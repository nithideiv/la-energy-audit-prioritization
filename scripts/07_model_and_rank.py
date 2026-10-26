from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config

BASE_NUM = ["floor_area", "building_age", "latitude", "longitude", "year"]
WEATHER = ["mean_temp_f", "total_prcp_mm", "hdd65", "cdd65", "hot_days_90f"]
CAT = ["property_type"]


def pipeline(model, numeric_cols):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def grouped_oof(df, features, model, splits):
    pred = np.full(len(df), np.nan)
    cv = GroupKFold(n_splits=min(splits, df["building_id"].nunique()))
    for train, test in cv.split(df, groups=df["building_id"]):
        p = pipeline(model(), features)
        p.fit(df.iloc[train][features + CAT], df.iloc[train]["site_eui"])
        pred[test] = p.predict(df.iloc[test][features + CAT])
    return pred


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5, "R2": r2_score(y, p)}


def main():
    cfg = load_config()
    df = pd.read_csv(ROOT / "data/processed/model_table.csv", dtype={"building_id": str})
    extreme = df[df["site_eui"] > cfg["site_eui_model_max"]].copy()
    extreme.to_csv(ROOT / "outputs/extreme_site_eui_review.csv", index=False)
    df = df[
        df["site_eui"].notna()
        & df["floor_area"].gt(0)
        & df["station_id"].notna()
        & df["mean_temp_f"].notna()
        & df["site_eui"].le(cfg["site_eui_model_max"])
    ].reset_index(drop=True)
    print(f"Modeling {len(df):,} rows; set aside {len(extreme):,} rows above the Site EUI review threshold")
    specs = {
        "linear_building": (BASE_NUM, lambda: LinearRegression()),
        "linear_weather": (BASE_NUM + WEATHER, lambda: LinearRegression()),
        "rf_building": (BASE_NUM, lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=cfg["random_state"])),
        "rf_weather": (BASE_NUM + WEATHER, lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=cfg["random_state"])),
    }
    rows = []
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=cfg["random_state"])
    train_idx, test_idx = next(splitter.split(df, groups=df["building_id"]))
    for name, (features, factory) in specs.items():
        heldout_model = pipeline(factory(), features)
        heldout_model.fit(df.iloc[train_idx][features + CAT], df.iloc[train_idx]["site_eui"])
        heldout_pred = heldout_model.predict(df.iloc[test_idx][features + CAT])
        rows.append({"model": name, "scope": "fixed_20pct_heldout_buildings", **metrics(df.iloc[test_idx]["site_eui"], heldout_pred)})
        p = grouped_oof(df, features, factory, cfg["n_splits"])
        df[f"pred_{name}"] = p
        rows.append({"model": name, "scope": "grouped_oof_all_buildings", **metrics(df["site_eui"], p)})
    # Baselines are fold-specific and therefore leakage-safe.
    for baseline in ["overall_median", "property_type_median"]:
        p = np.full(len(df), np.nan)
        cv = GroupKFold(n_splits=min(cfg["n_splits"], df["building_id"].nunique()))
        for train, test in cv.split(df, groups=df["building_id"]):
            tr, te = df.iloc[train], df.iloc[test]
            if baseline == "overall_median":
                p[test] = tr["site_eui"].median()
            else:
                med = tr.groupby("property_type")["site_eui"].median()
                p[test] = te["property_type"].map(med).fillna(tr["site_eui"].median())
        rows.append({"model": baseline, "scope": "grouped_oof_all_buildings", **metrics(df["site_eui"], p)})
    pd.DataFrame(rows).sort_values("MAE").to_csv(ROOT / "outputs/model_metrics.csv", index=False)
    chosen = "rf_weather"
    df["predicted_site_eui"] = df[f"pred_{chosen}"]
    df["eui_residual"] = df["site_eui"] - df["predicted_site_eui"]
    df["excess_kbtu"] = df["eui_residual"] * df["floor_area"]
    df.to_csv(ROOT / "outputs/oof_predictions.csv", index=False)
    per = df.groupby("building_id").agg(
        years=("year", "nunique"), positive_years=("eui_residual", lambda x: int((x > 0).sum())),
        mean_annual_excess_kbtu=("excess_kbtu", "mean"), mean_actual_site_eui=("site_eui", "mean"),
        mean_predicted_site_eui=("predicted_site_eui", "mean"), property_type=("property_type", "first"),
        address=("address", "first"), floor_area=("floor_area", "median"), emissions=("emissions", "mean"),
    ).reset_index()
    eligible = per[(per["years"] == len(cfg["years"])) & (per["positive_years"] >= 2)].copy()
    eligible = eligible.sort_values("mean_annual_excess_kbtu", ascending=False)
    # Does not change the ranking order; flags extreme actual-vs-predicted gaps for data review.
    eligible["actual_to_predicted_ratio"] = eligible["mean_actual_site_eui"] / eligible["mean_predicted_site_eui"]
    eligible["check_data_first"] = eligible["actual_to_predicted_ratio"] > 5
    eligible.insert(0, "priority_rank", range(1, len(eligible) + 1))
    eligible.to_csv(ROOT / "outputs/audit_priority_ranking.csv", index=False)
    top_sets = {}
    for name in specs:
        residual = df["site_eui"] - df[f"pred_{name}"]
        temp = df.assign(_residual=residual, _excess=residual * df["floor_area"])
        agg = temp.groupby("building_id").agg(years=("year", "nunique"), positive=("_residual", lambda x: (x > 0).sum()), score=("_excess", "mean"))
        agg = agg[(agg["years"] == len(cfg["years"])) & (agg["positive"] >= 2)].sort_values("score", ascending=False)
        n = max(1, int(np.ceil(len(agg) * .10))) if len(agg) else 0
        top_sets[name] = set(agg.head(n).index)
    sensitivity = []
    names = list(top_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = top_sets[a] | top_sets[b]
            sensitivity.append({"model_a": a, "model_b": b, "top_decile_jaccard": len(top_sets[a] & top_sets[b]) / len(union) if union else np.nan})
    pd.DataFrame(sensitivity).to_csv(ROOT / "outputs/model_ranking_stability.csv", index=False)
    print(pd.DataFrame(rows).sort_values("MAE").to_string(index=False))
    print(f"Ranked {len(eligible):,} eligible buildings")


if __name__ == "__main__":
    main()

