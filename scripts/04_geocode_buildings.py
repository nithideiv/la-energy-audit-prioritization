from pathlib import Path
import argparse
import csv
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import write_json


def prepare():
    df = pd.read_csv(ROOT / "data/processed/buildings_clean.csv", dtype={"building_id": str})
    missing = df[df["latitude"].isna() | df["longitude"].isna()][["building_id", "address"]].drop_duplicates("building_id")
    if "postal_code" in df.columns:
        zips = df[["building_id", "postal_code"]].drop_duplicates("building_id")
        missing = missing.merge(zips, on="building_id", how="left")
    else:
        missing["postal_code"] = ""
    existing_path = ROOT / "data/interim/geocodes.csv"
    if existing_path.exists():
        existing_ids = set(pd.read_csv(existing_path, dtype={"building_id": str})["building_id"])
        missing = missing[~missing["building_id"].isin(existing_ids)]
    # Census batch format: unique ID, street, city, state, ZIP. ZIP may be blank.
    out = pd.DataFrame({"id": missing["building_id"], "street": missing["address"], "city": "Los Angeles", "state": "CA", "zip": missing["postal_code"].fillna("")})
    path = ROOT / "data/interim/census_batch_input.csv"
    out.to_csv(path, index=False, header=False)
    print(f"Prepared {len(out):,} unique addresses at {path}")


def parse(path: Path):
    # Census currently returns more than one layout. Some result files contain
    # one combined input-address field; others repeat street/city/state/ZIP.
    # Locate Match/No_Match instead of assuming a fixed number of columns.
    records = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_number, row in enumerate(csv.reader(f), start=1):
            status_index = next((i for i, value in enumerate(row) if value.strip().lower() in {"match", "no_match", "no match", "tie"}), None)
            if status_index is None:
                raise ValueError(f"Could not find Census match status on line {line_number}: {row[:6]}")
            tail = row[status_index:]
            coordinates = tail[3].strip() if len(tail) > 3 else ""
            trailing_start = 4
            # Be tolerant of an unquoted longitude,latitude pair.
            if len(tail) > 4 and "," not in coordinates:
                try:
                    float(coordinates)
                    float(tail[4])
                    coordinates = f"{coordinates},{tail[4].strip()}"
                    trailing_start = 5
                except ValueError:
                    pass
            records.append({
                "building_id": str(row[0]).strip(),
                "input_address": ", ".join(part.strip() for part in row[1:status_index]),
                "match": tail[0].strip(),
                "match_type": tail[1].strip() if len(tail) > 1 else "",
                "matched_address": tail[2].strip() if len(tail) > 2 else "",
                "coordinates": coordinates,
                "tigerline": tail[trailing_start].strip() if len(tail) > trailing_start else "",
                "side": tail[trailing_start + 1].strip() if len(tail) > trailing_start + 1 else "",
            })
    geo = pd.DataFrame(records)
    coords = geo["coordinates"].str.split(",", expand=True)
    geo["longitude"] = pd.to_numeric(coords[0], errors="coerce")
    geo["latitude"] = pd.to_numeric(coords[1], errors="coerce")
    matched = geo["match"].astype(str).str.lower().eq("match") & geo["latitude"].notna()
    new_matches = geo.loc[matched, ["building_id", "latitude", "longitude", "match_type", "matched_address"]]
    out_path = ROOT / "data/interim/geocodes.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype={"building_id": str})
        combined = pd.concat([existing, new_matches], ignore_index=True).drop_duplicates("building_id", keep="last")
    else:
        combined = new_matches
    combined.to_csv(out_path, index=False)
    total_buildings = pd.read_csv(ROOT / "data/processed/buildings_clean.csv", dtype={"building_id": str})["building_id"].nunique()
    write_json(ROOT / "outputs/geocoding_report.json", {"latest_batch_submitted": len(geo), "latest_batch_matched": int(matched.sum()), "cumulative_matched": len(combined), "total_buildings": total_buildings, "cumulative_match_rate": len(combined) / total_buildings if total_buildings else None})
    print(f"Latest batch matched {matched.sum():,} of {len(geo):,}; cumulative matches: {len(combined):,} of {total_buildings:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--parse", type=Path)
    args = p.parse_args()
    prepare() if args.prepare else parse(args.parse)

