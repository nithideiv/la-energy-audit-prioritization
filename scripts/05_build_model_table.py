from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import haversine_miles, load_config, write_json


def building_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag, but do not overwrite, changing building attributes across years."""
    grouped = df.groupby("building_id", dropna=False)
    flags = grouped.agg(
        years_present=("year", "nunique"),
        property_type_values=("property_type", "nunique"),
        year_built_values=("year_built", "nunique"),
        floor_area_min=("floor_area", "min"),
        floor_area_max=("floor_area", "max"),
        max_number_of_buildings=("number_of_buildings", "max"),
    ).reset_index()
    flags["floor_area_change_ratio"] = flags["floor_area_max"] / flags["floor_area_min"]
    flags["property_type_changed"] = flags["property_type_values"].gt(1)
    flags["year_built_changed"] = flags["year_built_values"].gt(1)
    flags["floor_area_changed_over_20pct"] = flags["floor_area_change_ratio"].gt(1.2)
    flags["reported_multiple_buildings"] = flags["max_number_of_buildings"].gt(1)
    flags["implausibly_small_floor_area"] = flags["floor_area_min"].lt(1000)
    flags["needs_data_review"] = flags[[
        "property_type_changed", "year_built_changed", "floor_area_changed_over_20pct",
        "reported_multiple_buildings", "implausibly_small_floor_area",
    ]].any(axis=1)
    return flags


def main():
    cfg = load_config()
    b = pd.read_csv(ROOT / "data/processed/buildings_clean.csv", dtype={"building_id": str})
    geo_path = ROOT / "data/interim/geocodes.csv"
    if geo_path.exists():
        g = pd.read_csv(geo_path, dtype={"building_id": str})
        b = b.merge(g[["building_id", "latitude", "longitude"]], on="building_id", how="left", suffixes=("", "_geo"))
        b["latitude"] = b["latitude"].fillna(b["latitude_geo"])
        b["longitude"] = b["longitude"].fillna(b["longitude_geo"])
        b = b.drop(columns=["latitude_geo", "longitude_geo"])
    stations = cfg["stations"]
    distances = {sid: haversine_miles(b["latitude"], b["longitude"], s["latitude"], s["longitude"]) for sid, s in stations.items()}
    dist = pd.DataFrame(distances, index=b.index)
    located = b["latitude"].notna() & b["longitude"].notna()
    b["station_id"] = pd.NA
    b["station_distance_miles"] = np.nan
    b.loc[located, "station_id"] = dist.loc[located].idxmin(axis=1)
    b.loc[located, "station_distance_miles"] = dist.loc[located].min(axis=1)
    w = pd.read_csv(ROOT / "data/processed/weather_annual.csv")
    model = b.merge(w, left_on=["station_id", "year"], right_on=["STATION", "year"], how="left")
    model["building_age"] = model["year"] - model["year_built"]
    model.loc[model["building_age"] < 0, "building_age"] = np.nan
    flags = building_quality_flags(model)
    flags.to_csv(ROOT / "outputs/building_consistency_flags.csv", index=False)
    model = model.merge(flags, on="building_id", how="left", validate="many_to_one")
    model.to_csv(ROOT / "data/processed/model_table.csv", index=False)
    report = {"building_year_rows": len(model), "geocoded_rows": int(model["station_id"].notna().sum()), "weather_joined_rows": int(model["mean_temp_f"].notna().sum()), "median_station_distance_miles": float(model["station_distance_miles"].median()), "buildings_needing_data_review": int(flags["needs_data_review"].sum())}
    write_json(ROOT / "outputs/join_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

