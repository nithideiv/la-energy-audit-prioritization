from pathlib import Path
import sys
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config


def main():
    cfg = load_config()
    raw_dir = ROOT / "data/raw/weather"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for station in cfg["stations"]:
        url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station}.csv"
        path = raw_dir / f"{station}.csv"
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        df = pd.read_csv(path, low_memory=False)
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df[df["DATE"].dt.year.isin(cfg["years"])].copy()
        frames.append(df)
        print(f"{station}: retained {len(df):,} station-days")
    # Consolidate the downloaded frames before adding derived columns. This
    # prevents pandas fragmentation warnings caused by the wide NOAA schema.
    daily = pd.concat(frames, ignore_index=True).copy()
    daily.to_csv(ROOT / "data/interim/weather_daily.csv", index=False)

    # GHCN access CSV temperatures are tenths of °C; precipitation is tenths of mm.
    for c in ["TMAX", "TMIN", "PRCP"]:
        daily[c] = pd.to_numeric(daily.get(c), errors="coerce")
        # *_ATTRIBUTES is MFLAG,QFLAG,SFLAG. A nonblank QFLAG means NOAA flagged quality.
        attr = f"{c}_ATTRIBUTES"
        if attr in daily:
            qflag = daily[attr].fillna("").astype(str).str.split(",").str[1].fillna("").str.strip()
            rejected = int(qflag.ne("").sum())
            daily.loc[qflag.ne(""), c] = pd.NA
            print(f"{c}: excluded {rejected:,} quality-flagged observations")
    daily["tmax_f"] = daily["TMAX"] / 10 * 9 / 5 + 32
    daily["tmin_f"] = daily["TMIN"] / 10 * 9 / 5 + 32
    daily["tavg_f"] = (daily["tmax_f"] + daily["tmin_f"]) / 2
    daily["prcp_mm"] = daily["PRCP"] / 10
    daily["hdd65"] = (cfg["base_temperature_f"] - daily["tavg_f"]).clip(lower=0)
    daily["cdd65"] = (daily["tavg_f"] - cfg["base_temperature_f"]).clip(lower=0)
    daily["hot_day_90f"] = daily["tmax_f"] > cfg["hot_day_threshold_f"]
    daily["year"] = daily["DATE"].dt.year
    annual = daily.groupby(["STATION", "year"], as_index=False).agg(
        mean_temp_f=("tavg_f", "mean"),
        mean_tmax_f=("tmax_f", "mean"),
        mean_tmin_f=("tmin_f", "mean"),
        total_prcp_mm=("prcp_mm", "sum"),
        hdd65=("hdd65", "sum"),
        cdd65=("cdd65", "sum"),
        hot_days_90f=("hot_day_90f", "sum"),
        valid_temp_days=("tavg_f", "count"),
        valid_prcp_days=("prcp_mm", "count"),
    )
    annual.to_csv(ROOT / "data/processed/weather_annual.csv", index=False)
    print(annual.to_string(index=False))


if __name__ == "__main__":
    main()

