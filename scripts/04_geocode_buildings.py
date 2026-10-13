from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import write_json


def prepare():
    df = pd.read_csv(ROOT / "data/processed/buildings_clean.csv", dtype={"building_id": str})
    missing = df[df["latitude"].isna() | df["longitude"].isna()][["building_id", "address"]].drop_duplicates("building_id")
    # Census batch format: unique ID, street, city, state, ZIP. ZIP may be blank.
    out = pd.DataFrame({"id": missing["building_id"], "street": missing["address"], "city": "Los Angeles", "state": "CA", "zip": ""})
    path = ROOT / "data/interim/census_batch_input.csv"
    out.to_csv(path, index=False, header=False)
    print(f"Prepared {len(out):,} unique addresses at {path}")


def parse(path: Path):
    # Census batch output columns: id,input_address,match,match_type,matched_address,coordinates,tigerline,side
    geo = pd.read_csv(path, header=None, names=["building_id", "input_address", "match", "match_type", "matched_address", "coordinates", "tigerline", "side"], dtype={"building_id": str})
    coords = geo["coordinates"].str.split(",", expand=True)
    geo["longitude"] = pd.to_numeric(coords[0], errors="coerce")
    geo["latitude"] = pd.to_numeric(coords[1], errors="coerce")
    matched = geo["match"].astype(str).str.lower().eq("match") & geo["latitude"].notna()
    geo.loc[matched, ["building_id", "latitude", "longitude", "match_type", "matched_address"]].to_csv(ROOT / "data/interim/geocodes.csv", index=False)
    write_json(ROOT / "outputs/geocoding_report.json", {"submitted": len(geo), "matched": int(matched.sum()), "match_rate": float(matched.mean()) if len(geo) else None})
    print(f"Matched {matched.sum():,} of {len(geo):,} addresses")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--parse", type=Path)
    args = p.parse_args()
    prepare() if args.prepare else parse(args.parse)

