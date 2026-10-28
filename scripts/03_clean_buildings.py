from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import find_column, load_config, normalize_columns, numeric, write_json

ALIASES = {
    "building_id": ["building_id", "property_id", "property_gfa_id"],
    "address": ["building_address", "property_address", "address"],
    "year": ["program_year", "compliance_year", "reporting_year", "year"],
    "property_type": ["primary_property_type", "property_type"],
    "postal_code": ["postal_code", "zip_code", "zip"],
    "year_built": ["year_built"],
    "floor_area": ["gross_building_floor_area_ft", "property_gfa_calculated_buildings_and_parking_ft", "property_gfa_self_reported_ft", "gross_floor_area", "floor_area"],
    "site_eui": ["site_energy_use_intensity_eui_kbtu_ft", "site_eui_kbtu_ft", "site_eui", "site_energy_use_intensity_eui"],
    "emissions": ["total_ghg_emissions_metric_tons_co2e", "carbon_dioxide_emissions_metric_ton_co2e", "ghg_emissions"],
    # Self-reported % of the building occupied (0-100); blank for buildings that never filed energy data.
    "occupancy": ["occupancy", "occupancy_percent", "occupancy_pct"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
}


def clean(raw: pd.DataFrame, years: list[int]):
    raw = normalize_columns(raw)
    mapped = {}
    sources = {}
    for target, aliases in ALIASES.items():
        source = find_column(raw.columns, aliases, required=target not in {"emissions", "occupancy", "latitude", "longitude", "postal_code"})
        sources[target] = source
        mapped[target] = raw[source] if source else pd.NA
    df = pd.DataFrame(mapped)
    for c in ["year", "year_built", "floor_area", "site_eui", "emissions", "occupancy", "latitude", "longitude"]:
        df[c] = numeric(df[c])
    # Occupancy is a percentage; anything outside 0-100 is a reporting error, so treat it as missing.
    df.loc[~df["occupancy"].between(0, 100), "occupancy"] = np.nan
    df["building_id"] = df["building_id"].astype("string").str.strip()
    df["address"] = df["address"].astype("string").str.strip().str.upper()
    df["property_type"] = df["property_type"].astype("string").str.strip().replace("", pd.NA)
    df["postal_code"] = df["postal_code"].astype("string").str.extract(r"(\d{5})", expand=False)
    df = df[df["year"].isin(years)].copy()
    selected_year_rows = len(df)
    duplicate_rows = int(df.duplicated(["building_id", "year"], keep=False).sum())
    # Prefer the duplicate row with the most nonmissing core fields; retain one building-year.
    df["_complete"] = df[["address", "property_type", "year_built", "floor_area", "site_eui"]].notna().sum(axis=1)
    df = df.sort_values(["building_id", "year", "_complete"], ascending=[True, True, False]).drop_duplicates(["building_id", "year"]).drop(columns="_complete")
    invalid_area = df["floor_area"].le(0) | df["floor_area"].isna()
    invalid_eui = df["site_eui"].le(0) | df["site_eui"].isna()
    df["usable_target"] = ~(invalid_area | invalid_eui)
    usable = df[df["usable_target"]].copy()
    years_per = usable.groupby("building_id")["year"].nunique()
    report = {
        # Original portal field used for `year`. LADBS instructions treat Program Year N as
        # calendar-year-N consumption data, so `year` is joined to same-year weather.
        "year_source_column": sources["year"],
        "usable_rows_by_year": {int(k): int(v) for k, v in usable.groupby("year").size().items()},
        "raw_rows": len(raw), "selected_year_rows_before_dedup": selected_year_rows,
        "duplicate_rows_flagged": duplicate_rows, "rows_after_dedup": len(df),
        "invalid_or_missing_floor_area": int(invalid_area.sum()), "invalid_or_missing_site_eui": int(invalid_eui.sum()),
        "usable_rows": len(usable), "usable_unique_buildings": int(usable["building_id"].nunique()),
        "usable_buildings_all_three_years": int((years_per == len(years)).sum()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
    }
    return usable, report


def main():
    cfg = load_config()
    raw = pd.read_csv(ROOT / "data/raw/ebewe_all.csv", low_memory=False)
    clean_df, report = clean(raw, cfg["years"])
    out = ROOT / "data/processed/buildings_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out, index=False)
    write_json(ROOT / "outputs/data_quality_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

