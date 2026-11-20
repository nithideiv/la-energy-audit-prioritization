"""Per-property-type models versus the general model, and a model portfolio.

Why: one model for all 72 property types is dominated by multifamily housing (53%
of rows), and the per-type median baseline beats it on MAE. This script asks
whether fitting separate models for subsets of building types helps, and for
which types.

Three ways of splitting the data, each fit with a ridge and a random forest:
  mf_split   Multifamily Housing vs. everything else (2 buckets)
  use_group  Six hand-defined use categories (residential, office/services, ...)
  type_top   One model per property type with >= MIN_BLDGS buildings; every
             rarer type shares one pooled model (thin buckets are never fit alone)

All models, including the general linear and random-forest models, are scored on
one shared, deterministic 5-fold split in which all rows of a building stay in the
same fold, so every comparison is paired. The folds are built here rather than with
sklearn's GroupKFold, whose tie-breaking between equal-sized groups changes with the
sklearn/numpy version. As a result the general-model numbers here differ slightly
from outputs/model_metrics.csv (same models, different fold assignment).

Portfolio rule (per property type with enough buildings, plus one pooled
"all other types" segment): choose the simplest candidate whose building-level
bootstrap MAE difference from the best candidate includes zero. Complexity
order: type median < general linear < per-group ridge < general random forest
< per-group random forest.

Writes:
  outputs/per_type_oof_predictions.csv  out-of-fold prediction of every candidate
  outputs/per_type_metrics.csv          MAE/RMSE/R2/bias for every type x candidate
  outputs/grouping_comparison.csv       overall metrics for every grouping x family
  outputs/model_portfolio.csv           recommended model per segment, with CIs
  outputs/portfolio_summary.json        overall portfolio metrics and the takeaway
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config

OUT = ROOT / "outputs"
MIN_BLDGS = 50          # a property type gets its own model/segment at or above this many buildings
MIN_TRAIN_ROWS = 30     # fewer training rows than this in a group -> fall back to the general model
N_BOOT = 1000
RIDGE_ALPHA = 10.0

USE_GROUP = {
    "Residential": ["Multifamily Housing", "Senior Living Community", "Residential Care Facility",
                    "Residence Hall/Dormitory", "Other - Lodging/Residential", "Hotel"],
    "Office & services": ["Office", "Medical Office", "Financial Office", "Bank Branch", "Mixed Use Property",
                          "Other - Services", "Personal Services (Health/Beauty, Dry Cleaning, etc.)",
                          "Fitness Center/Health Club/Gym", "Veterinary Office", "Mailing Center/Post Office",
                          "Urgent Care/Clinic/Other Outpatient", "Outpatient Rehabilitation/Physical Therapy",
                          "Ambulatory Surgical Center"],
    "Retail & food": ["Retail Store", "Strip Mall", "Supermarket/Grocery Store", "Other - Mall", "Enclosed Mall",
                      "Lifestyle Center", "Wholesale Club/Supercenter", "Vehicle Dealership",
                      "Convenience Store without Gas Station", "Food Sales", "Food Service", "Restaurant",
                      "Other - Restaurant/Bar", "Bar/Nightclub", "Vehicle Repair Services",
                      "Repair Services (Vehicle, Shoe, Locksmith, etc.)"],
    "Industrial & storage": ["Non-Refrigerated Warehouse", "Distribution Center", "Manufacturing/Industrial Plant",
                             "Self-Storage Facility", "Refrigerated Warehouse", "Parking", "Energy/Power Station",
                             "Other - Utility", "Data Center"],
    "Public & institutional": ["Fire Station", "Police Station", "Worship Facility", "K-12 School",
                               "College/University", "Other - Public Services", "Other - Education",
                               "Vocational School", "Adult Education", "Pre-school/Daycare", "Courthouse",
                               "Community Center and Social Meeting Hall", "Social/Meeting Hall", "Museum",
                               "Performing Arts", "Movie Theater", "Convention Center", "Indoor Arena",
                               "Ice/Curling Rink", "Bowling Alley", "Other - Entertainment/Public Assembly",
                               "Other - Recreation", "Transportation Terminal/Station"],
    "Healthcare & lab": ["Hospital (General Medical & Surgical)", "Other - Specialty Hospital", "Laboratory",
                         "Other - Technology/Science"],
}
TYPE_TO_USE = {t: g for g, types in USE_GROUP.items() for t in types}

# Simplest -> most complex; used by the portfolio rule. Every model also has a "_log" twin that is
# trained on log(1 + Site EUI): Site EUI is heavy-tailed, and a model fit with squared error on the raw
# value is pulled toward the few huge buildings, which hurts MAE. The log twin predicts the typical
# building instead (back-transformed, clipped to [0, site_eui_model_max]).
SUFFIXES = ["", "_log"]
GROUPINGS = ["mf_split", "use_group", "type_top"]
COMPLEXITY = {"median": 0}
LABELS = {"median": "Property-type median"}
GROUP_NAMES = {"mf_split": "multifamily split", "use_group": "use-category", "type_top": "per-type"}
for sfx in SUFFIXES:
    tag = " (log target)" if sfx else ""
    COMPLEXITY[f"gen_linear{sfx}"] = 1
    COMPLEXITY[f"gen_rf{sfx}"] = 3
    LABELS[f"gen_linear{sfx}"] = f"General linear{tag}"
    LABELS[f"gen_rf{sfx}"] = f"General random forest{tag}"
    for g in GROUPINGS:
        COMPLEXITY[f"{g}_ridge{sfx}"] = 2
        COMPLEXITY[f"{g}_rf{sfx}"] = 4
        LABELS[f"{g}_ridge{sfx}"] = f"Ridge, {GROUP_NAMES[g]}{tag}"
        LABELS[f"{g}_rf{sfx}"] = f"Random forest, {GROUP_NAMES[g]}{tag}"
EUI_CAP = 2000.0


def _inverse_log(z):
    return np.clip(np.expm1(z), 0, EUI_CAP)


def log_target(factory):
    """Wrap a model factory so the model is trained on log(1 + Site EUI)."""
    return lambda: TransformedTargetRegressor(regressor=factory(), func=np.log1p, inverse_func=_inverse_log,
                                              check_inverse=False)


def load_07():
    """Reuse the preprocessing pipeline and feature lists of the main modeling script."""
    spec = importlib.util.spec_from_file_location("model_and_rank", ROOT / "scripts" / "07_model_and_rank.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics_row(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    out = {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5,
           "median_AE": float(np.median(np.abs(y - p))), "bias": float(np.mean(y - p))}
    out["R2"] = r2_score(y, p) if len(y) > 1 and np.var(y) > 0 else np.nan
    return out


def make_folds(building_ids, n_splits):
    """Balanced, deterministic grouped folds: largest buildings first, each to the lightest fold."""
    _, inverse = np.unique(building_ids, return_inverse=True)
    sizes = np.bincount(inverse)
    load = np.zeros(n_splits)
    group_fold = np.zeros(len(sizes), dtype=int)
    for g in np.argsort(-sizes, kind="stable"):
        k = int(np.argmin(load))
        load[k] += sizes[g]
        group_fold[g] = k
    return group_fold[inverse]


def assign_groupings(df, bldgs_per_type):
    big = bldgs_per_type[bldgs_per_type >= MIN_BLDGS].index
    return {
        "mf_split": np.where(df["property_type"] == "Multifamily Housing", "Multifamily", "Non-multifamily"),
        "use_group": df["property_type"].map(TYPE_TO_USE).fillna("Other / unclassified").to_numpy(),
        "type_top": np.where(df["property_type"].isin(big), df["property_type"], "Pooled other types"),
    }


def grouped_fit_predict(df, folds, group, features, factory, fallback, pipeline_fn, n_folds):
    """Fit one model per group inside each training fold; small groups use the fallback."""
    pred = np.full(len(df), np.nan)
    y = df["site_eui"].to_numpy()
    cols = features + ["property_type"]
    for f in range(n_folds):
        test_idx = np.where(folds == f)[0]
        train_idx = np.where(folds != f)[0]
        for g in np.unique(group[test_idx]):
            te = test_idx[group[test_idx] == g]
            tr = train_idx[group[train_idx] == g]
            if len(tr) < MIN_TRAIN_ROWS:
                pred[te] = fallback[te]
                continue
            model = pipeline_fn(factory(), features)
            model.fit(df.iloc[tr][cols], y[tr])
            pred[te] = model.predict(df.iloc[te][cols])
    return pred


def paired_bootstrap(abs_a, abs_b, bldg, rng):
    """Building-level bootstrap of MAE(a) - MAE(b). Returns (diff, ci_low, ci_high)."""
    d = pd.Series(np.asarray(abs_a) - np.asarray(abs_b))
    g = d.groupby(np.asarray(bldg)).agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    est = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def fit_candidates(df, cand, folds, features, m07, seed, n_folds, bldgs_per_type):
    """Fit every general and grouped model out of fold and add its predictions to `cand`."""
    y = df["site_eui"].to_numpy()
    one_group = np.zeros(len(df), dtype=int)
    no_fallback = np.full(len(df), np.nan)
    forest = lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    ridge = lambda: Ridge(alpha=RIDGE_ALPHA)
    for name, factory in {"gen_linear": LinearRegression, "gen_rf": forest}.items():
        for sfx in SUFFIXES:
            t = time.time()
            wrapped = log_target(factory) if sfx else factory
            cand[f"{name}{sfx}"] = grouped_fit_predict(df, folds, one_group, features, wrapped, no_fallback,
                                                       m07.pipeline, n_folds)
            print(f"  {name}{sfx}: MAE {mean_absolute_error(y, cand[f'{name}{sfx}']):.2f}  ({time.time() - t:.0f}s)")
    groups = assign_groupings(df, bldgs_per_type)
    for g, labels in groups.items():
        print(f"{g}: {pd.Series(labels).nunique()} groups; smallest has {pd.Series(labels).value_counts().min()} rows")
    for g in GROUPINGS:
        for fam, factory, gen_name in [("ridge", ridge, "gen_linear"), ("rf", forest, "gen_rf")]:
            for sfx in SUFFIXES:
                t = time.time()
                wrapped = log_target(factory) if sfx else factory
                col = f"{g}_{fam}{sfx}"
                cand[col] = grouped_fit_predict(df, folds, groups[g], features, wrapped,
                                                cand[f"{gen_name}{sfx}"].to_numpy(), m07.pipeline, n_folds)
                print(f"  {col}: MAE {mean_absolute_error(y, cand[col]):.2f}  ({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse saved out-of-fold predictions instead of refitting (reruns only the portfolio step)")
    reuse = ap.parse_args().reuse
    t0 = time.time()
    cfg = load_config()
    m07 = load_07()
    features = m07.BASE_NUM + m07.WEATHER
    seed = cfg["random_state"]
    n_folds = cfg["n_splits"]

    # Same modeled rows as 07 (it saved them, with its own out-of-fold predictions, in oof_predictions.csv).
    df = pd.read_csv(OUT / "oof_predictions.csv", dtype={"building_id": str})
    y = df["site_eui"].to_numpy()

    # Deterministic building-grouped folds. sklearn's GroupKFold breaks ties between equal-sized groups
    # in a way that depends on the sklearn/numpy version, so folds are built here instead. Every
    # candidate below, including the general models, is scored on these folds so comparisons are paired.
    folds = make_folds(df["building_id"].to_numpy(), n_folds)
    df["fold"] = folds
    print("Rows per fold:", np.bincount(folds).tolist())

    # Fold-specific property-type median baseline.
    med = np.full(len(df), np.nan)
    for k in range(n_folds):
        tr, te = df[folds != k], df[folds == k]
        by_type = tr.groupby("property_type")["site_eui"].median()
        med[folds == k] = te["property_type"].map(by_type).fillna(tr["site_eui"].median()).to_numpy()

    cand = pd.DataFrame({
        "building_id": df["building_id"], "year": df["year"], "property_type": df["property_type"],
        "site_eui": y, "fold": folds, "median": med,
    })
    models = list(COMPLEXITY)
    bldgs_per_type = df.groupby("property_type")["building_id"].nunique()
    cache = OUT / "per_type_oof_predictions.csv"
    saved = pd.read_csv(cache, dtype={"building_id": str}) if (reuse and cache.exists()) else None
    if saved is not None and set(models) <= set(saved.columns) and (saved["fold"].to_numpy() == folds).all():
        for m in models[1:]:
            cand[m] = saved[m].to_numpy()
        print("Reused saved out-of-fold predictions; only the portfolio step is rerun.")
    else:
        fit_candidates(df, cand, folds, features, m07, seed, n_folds, bldgs_per_type)
        cand.to_csv(cache, index=False)
    stored = pd.read_csv(OUT / "model_metrics.csv")
    stored = stored[stored["scope"] == "grouped_oof_all_buildings"].set_index("model")["MAE"]
    print("General models are scored on the shared folds. For reference, 07 stored on its own folds: "
          f"linear_weather {stored['linear_weather']:.2f}, rf_weather {stored['rf_weather']:.2f}, "
          f"property_type_median {stored['property_type_median']:.2f} "
          f"(shared-fold property-type median: {mean_absolute_error(y, med):.2f})")

    # Reference models for "general" and "per-type" in the comparison plots: the best of each kind by MAE.
    ref_general = min([m for m in models if m.startswith("gen_")], key=lambda m: mean_absolute_error(y, cand[m]))
    ref_per_type = min([m for m in models if m.startswith("type_top_")], key=lambda m: mean_absolute_error(y, cand[m]))
    print(f"Reference general model: {ref_general}; reference per-type model: {ref_per_type}")

    # ---- overall comparison of every grouping x family -------------------------------------------
    overall = pd.DataFrame([{"model": m, "label": LABELS[m], **metrics_row(y, cand[m])} for m in models]
                           ).sort_values("MAE").reset_index(drop=True)
    overall.to_csv(OUT / "grouping_comparison.csv", index=False)

    # ---- per-type error for every type (small n flagged, never silently dropped) -----------------
    rows = []
    for t, sub in cand.groupby("property_type"):
        n_b = sub["building_id"].nunique()
        for m in models:
            rows.append({"property_type": t, "model": m, "n_rows": len(sub), "n_buildings": n_b,
                         "low_n": n_b < MIN_BLDGS, "mean_site_eui": sub["site_eui"].mean(),
                         **metrics_row(sub["site_eui"], sub[m])})
    per_type = pd.DataFrame(rows)
    per_type["MAE_pct_of_mean_eui"] = 100 * per_type["MAE"] / per_type["mean_site_eui"]
    per_type.to_csv(OUT / "per_type_metrics.csv", index=False)

    # ---- portfolio: choose a model per segment ---------------------------------------------------
    rng = np.random.default_rng(seed)
    big = bldgs_per_type[bldgs_per_type >= MIN_BLDGS].index
    segment = np.where(cand["property_type"].isin(big), cand["property_type"], "All other types")
    cand["segment"] = segment
    port_rows = []
    chosen_pred = np.full(len(cand), np.nan)
    for seg, sub in cand.groupby("segment"):
        abs_err = {m: np.abs(sub["site_eui"] - sub[m]).to_numpy() for m in models}
        maes = {m: abs_err[m].mean() for m in models}
        best = min(maes, key=maes.get)
        order = sorted(models, key=lambda m: (COMPLEXITY[m], maes[m]))
        pick = best
        for m in order:
            diff, lo, hi = paired_bootstrap(abs_err[m], abs_err[best], sub["building_id"], rng)
            if lo <= 0:           # not distinguishable from the best -> take the simpler model
                pick = m
                break
        d_gen = paired_bootstrap(abs_err[pick], abs_err[ref_general], sub["building_id"], rng)
        ml_models = [m for m in models if m != "median"]
        best_ml = min(ml_models, key=maes.get)
        d_med = paired_bootstrap(abs_err[best_ml], abs_err["median"], sub["building_id"], rng)
        chosen_pred[sub.index] = sub[pick].to_numpy()
        port_rows.append({
            "segment": seg, "n_rows": len(sub), "n_buildings": sub["building_id"].nunique(),
            "lowest_mae_model": best, "recommended_model": pick, "recommended_label": LABELS[pick],
            "mae_recommended": maes[pick], "mae_lowest": maes[best],
            "reference_general_model": ref_general, "mae_general_ref": maes[ref_general],
            "reference_per_type_model": ref_per_type, "mae_per_type_ref": maes[ref_per_type],
            "mae_type_median": maes["median"],
            "diff_vs_general_ref": d_gen[0], "diff_vs_general_ref_ci_low": d_gen[1], "diff_vs_general_ref_ci_high": d_gen[2],
            "significantly_better_than_general_ref": d_gen[2] < 0,
            "best_ml_model": best_ml, "best_ml_label": LABELS[best_ml], "mae_best_ml": maes[best_ml],
            "diff_best_ml_vs_median": d_med[0], "diff_best_ml_vs_median_ci_low": d_med[1],
            "diff_best_ml_vs_median_ci_high": d_med[2], "best_ml_significantly_better_than_median": d_med[2] < 0,
            **{f"mae_{m}": maes[m] for m in models},
        })
    portfolio = pd.DataFrame(port_rows).sort_values("n_rows", ascending=False).reset_index(drop=True)
    portfolio.to_csv(OUT / "model_portfolio.csv", index=False)
    cand["portfolio_pred"] = chosen_pred
    cand["gen_ref"] = cand[ref_general]
    cand["per_type_ref"] = cand[ref_per_type]
    cand.to_csv(OUT / "per_type_oof_predictions.csv", index=False)

    port_all = metrics_row(y, chosen_pred)
    summary = {
        "n_rows": len(cand), "n_buildings": int(cand["building_id"].nunique()),
        "min_buildings_for_own_segment": MIN_BLDGS,
        "reference_general_model": ref_general, "reference_per_type_model": ref_per_type,
        "overall": {"portfolio": port_all, "general_reference": metrics_row(y, cand[ref_general]),
                    "per_type_reference": metrics_row(y, cand[ref_per_type]),
                    "property_type_median": metrics_row(y, cand["median"])},
        "portfolio_vs_general_ref_mae_diff_ci": dict(zip(["diff", "ci_low", "ci_high"], paired_bootstrap(
            np.abs(y - chosen_pred), np.abs(y - cand[ref_general]), cand["building_id"], rng))),
        "portfolio_vs_type_median_mae_diff_ci": dict(zip(["diff", "ci_low", "ci_high"], paired_bootstrap(
            np.abs(y - chosen_pred), np.abs(y - cand["median"]), cand["building_id"], rng))),
        "recommendations": portfolio[["segment", "recommended_model", "recommended_label", "n_buildings",
                                      "mae_recommended", "mae_general_ref", "mae_type_median",
                                      "significantly_better_than_general_ref", "best_ml_model", "mae_best_ml",
                                      "best_ml_significantly_better_than_median"]].to_dict("records"),
        "caveat": ("The portfolio is selected on the same out-of-fold predictions it is scored on, so its overall "
                   "MAE is slightly optimistic. Per-segment CIs are building-level bootstrap intervals."),
    }
    (OUT / "portfolio_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\nOverall grouped out-of-fold metrics")
    print(overall[["model", "MAE", "RMSE", "R2"]].round(3).to_string(index=False))
    print("\nPortfolio")
    print(portfolio[["segment", "n_buildings", "recommended_model", "mae_recommended", "mae_general_ref",
                     "mae_type_median", "best_ml_model", "mae_best_ml", "diff_best_ml_vs_median_ci_low",
                     "diff_best_ml_vs_median_ci_high"]].round(2).to_string(index=False))
    print(f"\nPortfolio overall: MAE {port_all['MAE']:.2f}, RMSE {port_all['RMSE']:.2f}, R2 {port_all['R2']:.3f}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
