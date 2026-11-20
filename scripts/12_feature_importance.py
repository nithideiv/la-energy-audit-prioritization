"""Feature importance for the report: which inputs drive the Site EUI prediction?

Method: grouped permutation importance on held-out data. For each fold, a random forest is
trained on the other folds; then one feature (or a group of features) is shuffled in the
held-out fold and the increase in MAE is recorded (kBtu/ft^2). Averaging over folds and
repeats gives a model-agnostic importance that is not biased toward high-cardinality or
continuous features the way impurity importance is. Folds are the shared deterministic
building-grouped folds from 11_per_type_models.py.

Feature groups: property_type is one feature (its one-hot columns are shuffled together);
the five weather variables are also reported as one "annual weather (5 vars)" group,
because they are strongly correlated and shuffling them one at a time understates them.

Writes:
  outputs/feature_importance_general.csv   importance in the general random forest
  outputs/feature_importance_by_type.csv   importance in a random forest trained within each large type
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config

OUT = ROOT / "outputs"
N_REPEATS = 3
TOP_TYPES = ["Multifamily Housing", "Office", "Non-Refrigerated Warehouse",
             "Manufacturing/Industrial Plant", "Distribution Center", "Retail Store"]
NICE = {"floor_area": "Floor area", "building_age": "Building age", "latitude": "Latitude",
        "longitude": "Longitude", "year": "Reporting year", "property_type": "Property type",
        "mean_temp_f": "Mean temp", "total_prcp_mm": "Precipitation", "hdd65": "Heating degree days",
        "cdd65": "Cooling degree days", "hot_days_90f": "Days above 90°F", "weather_group": "Annual weather (5 vars)"}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def permutation_importance(model, X_test, y_test, features, groups, rng):
    """Return {feature_or_group: mean MAE increase over N_REPEATS shuffles}."""
    base = mean_absolute_error(y_test, model.predict(X_test))
    out = {}
    for name, cols in {**{f: [f] for f in features}, **groups}.items():
        deltas = []
        for _ in range(N_REPEATS):
            Xp = X_test.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].iloc[perm].to_numpy()   # shuffle the columns jointly
            deltas.append(mean_absolute_error(y_test, model.predict(Xp)) - base)
        out[name] = float(np.mean(deltas))
    return base, out


def run(df, folds, feature_cols, group_defs, pipeline_fn, numeric, seed, label, rows_out, make_rf):
    rng = np.random.default_rng(seed)
    y = df["site_eui"].to_numpy()
    for k in sorted(np.unique(folds)):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        if len(tr) < 30 or len(te) < 5:
            continue
        model = pipeline_fn(make_rf(), numeric)
        model.fit(df.iloc[tr][feature_cols], y[tr])
        base, imp = permutation_importance(model, df.iloc[te][feature_cols], y[te], feature_cols, group_defs, rng)
        for feat, d in imp.items():
            rows_out.append({"scope": label, "fold": int(k), "feature": feat, "mae_increase": d,
                             "baseline_mae": base, "n_test": len(te)})


def summarize(raw):
    g = raw.groupby(["scope", "feature"]).agg(
        mae_increase=("mae_increase", "mean"), mae_increase_sd_across_folds=("mae_increase", "std"),
        baseline_mae=("baseline_mae", "mean"), folds=("fold", "nunique")).reset_index()
    g["pct_of_baseline_mae"] = 100 * g["mae_increase"] / g["baseline_mae"]
    g["label"] = g["feature"].map(NICE)
    g["is_group"] = g["feature"].eq("weather_group")
    return g.sort_values(["scope", "mae_increase"], ascending=[True, False])


def main():
    t0 = time.time()
    cfg = load_config()
    m07 = load_module("model_and_rank", "07_model_and_rank.py")
    m11 = load_module("per_type_models", "11_per_type_models.py")
    seed = cfg["random_state"]
    df = pd.read_csv(OUT / "oof_predictions.csv", dtype={"building_id": str})
    folds = m11.make_folds(df["building_id"].to_numpy(), cfg["n_splits"])
    features = m07.BASE_NUM + m07.WEATHER
    forest = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)

    # General model: all property types, property_type as a feature.
    raw = []
    cols = features + ["property_type"]
    run(df, folds, cols, {"weather_group": m07.WEATHER}, m07.pipeline, features, seed, "General model", raw, forest)
    general = summarize(pd.DataFrame(raw))
    general.drop(columns="scope").to_csv(OUT / "feature_importance_general.csv", index=False)
    print(f"General model done ({time.time() - t0:.0f}s)")
    print(general[["label", "mae_increase", "mae_increase_sd_across_folds", "pct_of_baseline_mae"]].round(3).to_string(index=False))

    # Within each large property type: train and test only on that type (no property_type feature).
    raw = []
    for t in TOP_TYPES:
        mask = (df["property_type"] == t).to_numpy()
        sub = df[mask].reset_index(drop=True)
        run(sub, folds[mask], features + ["property_type"], {"weather_group": m07.WEATHER},
            m07.pipeline, features, seed, t, raw, forest)
        print(f"  {t} done ({time.time() - t0:.0f}s)")
    by_type = summarize(pd.DataFrame(raw))
    by_type = by_type[by_type["feature"] != "property_type"]    # constant within a type
    by_type.to_csv(OUT / "feature_importance_by_type.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
