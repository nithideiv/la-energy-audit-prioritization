"""Run property-type experiments; writes outputs/property_type_results/ only.

Run steps 03 and 05 first. No downloads or external services are used.
"""
from pathlib import Path
import base64
import html
import json
import os
import sys

# Avoid oversubscribing the machine in boosting and numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config
from audit_reporting import file_sha256
from property_models import (BLOCKS, LABELS, MODEL_NAMES, assign_splits, cluster_interval,
                             cohort_masks, conformal_radius, fit_model, predict, route_predict)

OUT = ROOT / "outputs/property_type_results"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN = "#2b6cb0", "#dd7a1f", "#23856d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def score(frame, prediction):
    y = frame.site_eui.to_numpy()
    error = y - prediction
    per_type = pd.DataFrame({"type": frame.property_type, "ae": abs(error)}).groupby("type").ae.mean()
    mean_y = y.mean()
    return {"rows": len(frame), "buildings": frame.building_id.nunique(),
            "MAE": mean_absolute_error(y, prediction),
            "macro_type_MAE": per_type.mean(), "RMSE": mean_squared_error(y, prediction) ** .5,
            "R2": r2_score(y, prediction) if len(y) > 1 else np.nan,
            "mean_signed_error": error.mean(),
            "normalized_MAE_pct": 100 * abs(error).mean() / mean_y if mean_y > 0 else np.nan}


def plot_save(name):
    plt.savefig(FIG / f"{name}.png")
    plt.close()


def evaluate_cohort(name, frame, cfg):
    settings = cfg["property_type_analysis"]
    seed = cfg["random_state"]
    train = frame[frame.split.eq("train")]
    val = frame[frame.split.eq("validation")]
    cal = frame[frame.split.eq("calibration")]
    test = frame[frame.split.eq("test")].copy()
    general_models, selection, aggregate, type_rows = {}, [], [], []
    # All candidates see the same training, validation, and test buildings.
    for model_name in MODEL_NAMES:
        model = fit_model(model_name, train, seed, settings["forest_trees"])
        general_models[model_name] = model
        selection.append({"cohort": name, "property_type": "ALL", "scope": "general",
                          "model": model_name, **score(val, predict(model, val))})
    winner = min(MODEL_NAMES, key=lambda m: next(r["MAE"] for r in selection if r["model"] == m))
    general = general_models[winner]
    specialists, choices, chosen_specialist, recommendation = {}, {}, {}, {}
    for typ, train_type in train.groupby("property_type"):
        vt = val[val.property_type.eq(typ)]
        enough = (train_type.building_id.nunique() >= settings["min_specialist_train_buildings"]
                  and vt.building_id.nunique() >= settings["min_specialist_validation_buildings"])
        choices[typ] = "general"
        if not enough:
            recommendation[typ] = "General fallback: insufficient training/validation buildings"
            continue
        specialists[typ] = {}
        errors = {}
        for model_name in MODEL_NAMES:
            model = fit_model(model_name, train_type, seed, settings["forest_trees"])
            specialists[typ][model_name] = model
            errors[model_name] = mean_absolute_error(vt.site_eui, predict(model, vt))
            selection.append({"cohort": name, "property_type": typ, "scope": "specialist",
                              "model": model_name, **score(vt, predict(model, vt))})
        best = min(errors, key=errors.get)
        chosen_specialist[typ] = best
        general_error = mean_absolute_error(vt.site_eui, predict(general, vt))
        gain = (general_error - errors[best]) / general_error if general_error > 0 else 0
        if gain >= settings["min_validation_gain_fraction"]:
            choices[typ] = best
            recommendation[typ] = "Specialist: validation MAE improved by at least 5%"
        else:
            recommendation[typ] = "General fallback: specialist validation gain below 5%"
    selected_predict = lambda x: route_predict(x, general, specialists, choices)
    general_cal = predict(general, cal)
    portfolio_cal = selected_predict(cal)
    # Saved so scripts/12_property_type_extras.py can compare constant and type-scaled intervals.
    cal.assign(pred_general=general_cal, pred_portfolio=portfolio_cal, cohort=name).to_csv(
        OUT / f"calibration_predictions_{name}.csv", index=False)
    q_general = conformal_radius(cal, general_cal)
    q_portfolio = conformal_radius(cal, portfolio_cal)
    test["pred_general"] = predict(general, test)
    test["pred_portfolio"] = selected_predict(test)
    test["pred_specialist"] = np.nan
    test["selected_route"] = test.property_type.map(choices).fillna("general")
    test["general_model"] = winner
    test["interval_low"] = np.maximum(0, test.pred_portfolio - q_portfolio)
    test["interval_high"] = test.pred_portfolio + q_portfolio
    test["eui_residual"] = test.site_eui - test.pred_portfolio
    test["excess_kbtu"] = test.eui_residual * test.floor_area
    test["above_interval"] = test.site_eui > test.interval_high
    for model_name, model in general_models.items():
        p = predict(model, test)
        test[f"candidate_{model_name}"] = p
        aggregate.append({"cohort": name, "strategy": f"general_{model_name}",
                          "selected_general": model_name == winner, **score(test, p)})
    for strategy, p in [("selected_general", test.pred_general.to_numpy()),
                         ("selected_portfolio", test.pred_portfolio.to_numpy())]:
        aggregate.append({"cohort": name, "strategy": strategy,
                          "selected_general": False, **score(test, p)})
    for typ, group in test.groupby("property_type"):
        ix = group.index
        actual = group.site_eui
        gen = group.pred_general
        port = group.pred_portfolio
        specialist_mae = np.nan
        spec_low = spec_high = np.nan
        if typ in chosen_specialist:
            spec_p = predict(specialists[typ][chosen_specialist[typ]], group)
            test.loc[ix, "pred_specialist"] = spec_p
            specialist_mae = mean_absolute_error(actual, spec_p)
            spec_low, spec_high = cluster_interval(group, abs(actual - spec_p), seed)
        low, high = cluster_interval(group, abs(actual - gen), seed)
        diff_low, diff_high = cluster_interval(group, abs(actual - port) - abs(actual - gen), seed)
        covered = actual.between(group.interval_low, group.interval_high)
        types_train = train[train.property_type.eq(typ)]
        types_val = val[val.property_type.eq(typ)]
        general_val = mean_absolute_error(types_val.site_eui, predict(general, types_val)) if len(types_val) else np.nan
        specialist_val = (mean_absolute_error(types_val.site_eui, predict(specialists[typ][chosen_specialist[typ]], types_val))
                          if typ in chosen_specialist else np.nan)
        route = choices.get(typ, "general")
        type_rows.append({
            "cohort": name, "property_type": typ, "train_buildings": types_train.building_id.nunique(),
            "validation_buildings": types_val.building_id.nunique(), "test_buildings": group.building_id.nunique(),
            "test_rows": len(group), "general_model": winner,
            "best_specialist": chosen_specialist.get(typ, "not_fitted"),
            "validation_general_MAE": general_val, "validation_specialist_MAE": specialist_val,
            "selected_route": route, "selection_reason": recommendation.get(typ, "General fallback: type absent from training"),
            "general_MAE": mean_absolute_error(actual, gen), "general_MAE_low": low, "general_MAE_high": high,
            "specialist_MAE": specialist_mae, "specialist_MAE_low": spec_low, "specialist_MAE_high": spec_high,
            "portfolio_MAE": mean_absolute_error(actual, port),
            "portfolio_minus_general_MAE": (abs(actual - port) - abs(actual - gen)).mean(),
            "paired_diff_ci_low": diff_low, "paired_diff_ci_high": diff_high,
            "general_signed_error": (actual - gen).mean(), "portfolio_signed_error": (actual - port).mean(),
            "portfolio_normalized_MAE_pct": 100 * abs(actual - port).mean() / actual.mean(),
            "positive_residual_share": (actual > port).mean(),
            "interval_row_coverage": covered.mean(),
            "interval_building_coverage": covered.groupby(group.building_id).all().mean(),
            "limited_test_support": group.building_id.nunique() < settings["min_report_test_buildings"],
        })
    portfolio_abs = abs(test.site_eui - test.pred_portfolio)
    general_abs = abs(test.site_eui - test.pred_general)
    low, high = cluster_interval(test, portfolio_abs - general_abs, seed)
    baseline_abs = abs(test.site_eui - test.candidate_property_type_median)
    baseline_low, baseline_high = cluster_interval(test, general_abs - baseline_abs, seed)
    coverage = test.site_eui.between(test.interval_low, test.interval_high).groupby(test.building_id).all().mean()
    split_counts = frame.groupby("split").agg(rows=("year", "size"), buildings=("building_id", "nunique")).to_dict("index")
    summary = {"cohort": name, "selected_general": winner, "split_counts": split_counts,
               "specialist_types_selected": [k for k, v in choices.items() if v != "general"],
               "general_interval_radius": q_general, "portfolio_interval_radius": q_portfolio,
               "test_simultaneous_building_coverage": coverage,
               "portfolio_minus_general_test_MAE": (portfolio_abs - general_abs).mean(),
               "paired_difference_ci95": [low, high],
               "general_minus_baseline_test_MAE": (general_abs - baseline_abs).mean(),
               "general_minus_baseline_ci95": [baseline_low, baseline_high]}
    print(json.dumps(summary, default=float), flush=True)
    return test, pd.DataFrame(type_rows), pd.DataFrame(aggregate), pd.DataFrame(selection), summary, general, specialists, choices


def permutation_blocks(frame, prediction_fn, cohort, scope, seed, repeats=8, blocks=None):
    base = mean_absolute_error(frame.site_eui, prediction_fn(frame))
    rng = np.random.default_rng(seed)
    rows = []
    for label, columns in (BLOCKS if blocks is None else blocks).items():
        deltas = []
        for _ in range(repeats):
            permuted = frame.copy()
            order = rng.permutation(len(frame))
            permuted[columns] = frame.iloc[order][columns].to_numpy()
            deltas.append(mean_absolute_error(frame.site_eui, prediction_fn(permuted)) - base)
        rows.append({"cohort": cohort, "scope": scope, "feature_block": label,
                     "MAE_increase": np.mean(deltas), "repeat_std": np.std(deltas, ddof=1),
                     "repeats": repeats, "test_rows": len(frame), "baseline_MAE": base,
                     "test_buildings": frame.building_id.nunique()})
    return rows


def figures(test, types, aggregate, importance):
    shown = types[~types.limited_test_support].sort_values("test_buildings", ascending=False)
    # Two bars are genuine general-versus-specialist comparisons; absent fits stay absent.
    fig, ax = plt.subplots(figsize=(12, max(5, .42 * len(shown))))
    y = np.arange(len(shown))
    ax.barh(y - .18, shown.general_MAE, height=.34, color=BLUE, label="Validation-selected general model")
    available = shown.specialist_MAE.notna().to_numpy()
    ax.barh(y[available] + .18, shown.specialist_MAE[available], height=.34, color=ORANGE,
            label="Validation-selected specialist")
    ax.errorbar(shown.general_MAE, y - .18,
                xerr=[np.maximum(0, shown.general_MAE - shown.general_MAE_low),
                      np.maximum(0, shown.general_MAE_high - shown.general_MAE)], fmt="none", ecolor="#333", capsize=2)
    if available.any():
        a = shown.loc[available]
        ax.errorbar(a.specialist_MAE, y[available] + .18,
                    xerr=[np.maximum(0, a.specialist_MAE - a.specialist_MAE_low),
                          np.maximum(0, a.specialist_MAE_high - a.specialist_MAE)], fmt="none", ecolor="#333", capsize=2)
    ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(shown.property_type, shown.test_buildings)])
    ax.invert_yaxis()
    ax.set(xlabel="Held-out MAE (kBtu/ft²/year); lower is better",
           title="General versus per-property-type models · stable single-building cohort")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(.01, -.02, "n = test buildings. Bars use identical test rows. Whiskers: 95% building bootstrap. Missing orange bar: insufficient training/validation support.", fontsize=8)
    plot_save("01_general_vs_specialist")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = aggregate.cohort.unique()
    for offset, (strategy, label, color) in enumerate([
        ("general_property_type_median", "Property-type median baseline", "#8a8f98"),
        ("selected_general", "Selected general model", BLUE),
        ("selected_portfolio", "Selected portfolio", GREEN)]):
        sub = aggregate[aggregate.strategy.eq(strategy)].set_index("cohort").loc[names]
        for ax, metric in zip(axes, ["MAE", "macro_type_MAE"]):
            ax.bar(np.arange(len(names)) + (offset - 1) * .24, sub[metric], .24, color=color, label=label)
    for ax, title in zip(axes, ["Row-weighted error", "Equal-weight average of type errors"]):
        ax.set_xticks(np.arange(len(names)), ["All eligible", "Common types", "Stable single\nbuilding"])
        ax.set(ylabel="Test MAE (kBtu/ft²/year)", title=title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Three filtering experiments · each method uses the same rows within a cohort")
    fig.text(.02, -.01, "Cohorts contain different buildings; compare methods within each cohort. Macro error includes small types and is less stable.", fontsize=8)
    plot_save("02_filter_experiments")

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .34 * len(shown))))
    for ax, column, title in [(axes[0], "portfolio_signed_error", "Systematic under/overprediction"),
                               (axes[1], "portfolio_normalized_MAE_pct", "Error relative to each type's mean EUI")]:
        vals = shown[column]
        ax.barh(y, vals, color=[ORANGE if v > 0 else BLUE for v in vals])
        ax.set_yticks(y, shown.property_type)
        ax.invert_yaxis()
        ax.axvline(0, color="#555", lw=.8)
        ax.set_title(title)
    axes[0].set_xlabel("Mean actual − predicted EUI; positive = underprediction")
    axes[1].set_xlabel("100 × MAE / mean observed EUI (%)")
    fig.tight_layout()
    plot_save("03_property_type_fairness")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scope in zip(axes, ["general", "portfolio"]):
        data = importance[importance.scope.eq(scope)].sort_values("MAE_increase")
        ax.barh(data.feature_block, data.MAE_increase, color=BLUE if scope == "general" else GREEN,
                xerr=data.repeat_std, capsize=3)
        ax.axvline(0, color="#555", lw=.8)
        ax.set(title=f"Selected {scope}", xlabel="Test MAE increase after permutation")
    fig.suptitle("Feature-block permutation importance · larger means more predictive information")
    fig.tight_layout()
    plot_save("04_feature_importance")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    cap = np.quantile(np.r_[test.site_eui, test.pred_portfolio], .99)
    ax.scatter(test.site_eui, test.pred_portfolio, s=9, alpha=.22, color=BLUE)
    ax.plot([0, cap], [0, cap], "--", color="#555")
    ax.set(xlim=(0, cap), ylim=(0, cap), xlabel="Actual Site EUI", ylabel="Predicted Site EUI",
           title="Held-out actual versus predicted · selected portfolio")
    fig.text(.05, -.01, "Axes end at the combined 99th percentile for readability; metrics include all test rows.", fontsize=8)
    plot_save("05_actual_vs_predicted")

    candidates = aggregate[aggregate.cohort.eq("stable_single_building")
                           & aggregate.strategy.str.startswith("general_")].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [LABELS[v.removeprefix("general_")] for v in candidates.strategy]
    colors = [GREEN if v else BLUE for v in candidates.selected_general]
    ax.barh(labels, candidates.MAE, color=colors)
    for i, value in enumerate(candidates.MAE): ax.text(value + .15, i, f"{value:.2f}", va="center")
    ax.set(xlabel="Held-out MAE (kBtu/ft²/year)", title="Candidate general models · stable single-building cohort")
    ax.set_xlim(0, candidates.MAE.max() * 1.16)
    ax.invert_yaxis()
    fig.text(.02, -.01, "Green = selected using validation MAE. All bars use the same held-out buildings.", fontsize=8)
    plot_save("07_candidate_models")


def manager_example(test, types, cfg):
    grouped = test.groupby("building_id").agg(
        years=("year", "nunique"), positive_years=("eui_residual", lambda v: int((v > 0).sum())),
        mean_annual_excess_kbtu=("excess_kbtu", "mean"), mean_actual_site_eui=("site_eui", "mean"),
        mean_predicted_site_eui=("pred_portfolio", "mean"), property_type=("property_type", "first"),
        address=("address", "first"), selected_route=("selected_route", "first"),
        above_interval_years=("above_interval", "sum"))
    ranked = grouped[(grouped.years.eq(len(cfg["years"]))) & grouped.positive_years.ge(2)
                     & grouped.mean_annual_excess_kbtu.gt(0)].copy()
    ranked["actual_to_predicted_ratio"] = ranked.mean_actual_site_eui / ranked.mean_predicted_site_eui.replace(0, np.nan)
    ranked["check_data_first"] = ranked.actual_to_predicted_ratio.gt(5) | ranked.mean_predicted_site_eui.le(0)
    ranked = ranked.sort_values("mean_annual_excess_kbtu", ascending=False).reset_index()
    ranked.insert(0, "priority_rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(OUT / "heldout_audit_screening.csv", index=False)
    supported = set(types.loc[~types.limited_test_support, "property_type"])
    candidates = ranked[~ranked.check_data_first & ranked.property_type.isin(supported)]
    if candidates.empty:
        return None
    chosen = candidates.iloc[0]
    annual = test[test.building_id.eq(chosen.building_id)].sort_values("year").copy()
    annual["gap_pct"] = 100 * annual.eui_residual / annual.pred_portfolio.replace(0, np.nan)
    annual["actual_energy_kbtu"] = annual.site_eui * annual.floor_area
    annual["predicted_energy_kbtu"] = annual.pred_portfolio * annual.floor_area
    columns = ["building_id", "address", "year", "property_type", "floor_area", "occupancy",
               "site_eui", "pred_portfolio", "interval_low", "interval_high", "gap_pct",
               "actual_energy_kbtu", "predicted_energy_kbtu", "excess_kbtu", "above_interval", "selected_route"]
    annual[columns].to_csv(OUT / "building_manager_example.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(annual))
    axes[0].bar(x - .18, annual.site_eui, .35, color=ORANGE, label="Reported")
    axes[0].bar(x + .18, annual.pred_portfolio, .35, color=BLUE, label="Expected")
    axes[0].errorbar(x + .18, annual.pred_portfolio,
                     yerr=[annual.pred_portfolio - annual.interval_low,
                           annual.interval_high - annual.pred_portfolio], fmt="none", color="#333", capsize=4)
    axes[0].set(ylabel="Site EUI (kBtu/ft²/year)", title="Reported versus expected use")
    axes[0].legend()
    axes[1].bar(x, annual.excess_kbtu / 1e6, color=GREEN)
    axes[1].axhline(0, color="#555", lw=.8)
    axes[1].set(ylabel="Reported minus expected (million kBtu/year)", title="Consumption gap to investigate")
    for ax in axes:
        ax.set_xticks(x, annual.year.astype(int))
    fig.suptitle(f"Building {chosen.building_id} · {chosen.property_type}")
    fig.text(.02, -.02, "Whiskers: 90% marginal building-level interval, calibrated on separate buildings. Gaps are screening signals, not verified savings.", fontsize=8)
    fig.tight_layout()
    plot_save("06_building_manager_example")
    return {"building_id": chosen.building_id, "property_type": chosen.property_type,
            "positive_years": int(chosen.positive_years), "years": int(chosen.years),
            "above_interval_years": int(chosen.above_interval_years),
            "mean_annual_excess_kbtu": float(chosen.mean_annual_excess_kbtu),
            "selected_route": chosen.selected_route}


def write_report(summaries, aggregate, types, importance, example, cfg):
    primary = next(s for s in summaries if s["cohort"] == "stable_single_building")
    sel = aggregate[(aggregate.cohort.eq("stable_single_building")) & aggregate.strategy.isin(
        ["general_property_type_median", "selected_general", "selected_portfolio"])].copy()
    supported = types[~types.limited_test_support].copy().sort_values("test_buildings", ascending=False)
    table = supported[["property_type", "train_buildings", "validation_buildings", "test_buildings",
                       "general_MAE", "specialist_MAE", "portfolio_MAE", "selected_route"]]
    specialist_types = primary["specialist_types_selected"]
    takeaway = ("Validation selected type-specific median baselines for " + ", ".join(specialist_types) + ". "
                if specialist_types else "No specialist cleared the prespecified validation improvement threshold. ")
    # Describe the actual selected families, including future reruns with other winners.
    selected_rows = types[types.selected_route.ne("general")]
    if specialist_types:
        takeaway = "Validation-selected routes: " + "; ".join(
            f"{r.property_type}: {LABELS[r.selected_route]}" for r in selected_rows.itertuples()) + ". "
    takeaway += "All remaining types use the selected general model, including types with insufficient specialist training or validation support."
    global_mae = float(sel.loc[sel.strategy.eq("selected_general"), "MAE"].iloc[0])
    portfolio_mae = float(sel.loc[sel.strategy.eq("selected_portfolio"), "MAE"].iloc[0])
    baseline_mae = float(sel.loc[sel.strategy.eq("general_property_type_median"), "MAE"].iloc[0])
    lo, hi = primary["paired_difference_ci95"]
    paragraphs = [
        ("Main result", f"The selected general model is {LABELS[primary['selected_general']]}. "
         f"Held-out MAE is {global_mae:.2f} for that general model and {portfolio_mae:.2f} kBtu/ft²/year for the portfolio. "
         f"The paired portfolio-minus-general difference is {portfolio_mae-global_mae:.2f}, with a 95% building-bootstrap interval of [{lo:.2f}, {hi:.2f}]. "
         + ("The interval includes zero, so this test does not establish an overall improvement. " if lo <= 0 <= hi else
            "This interval describes held-out sampling uncertainty, conditional on the fitted models and selection split. ")
         + f"The property-type median baseline has MAE {baseline_mae:.2f}; the selected general model lowers that by "
         f"{100*(baseline_mae-global_mae)/baseline_mae:.1f}%. Its paired general-minus-baseline 95% interval is "
         f"[{primary['general_minus_baseline_ci95'][0]:.2f}, {primary['general_minus_baseline_ci95'][1]:.2f}]. " + takeaway),
        ("Design and filters", "All years of a building remain together. One fixed split allocates approximately 50% of buildings to training, "
         "20% to validation, 10% to interval calibration, and 20% to testing. The same building assignments are used in all three cohorts. "
         "The broad cohort requires valid positive area and EUI, joined weather, and EUI at or below the existing 2,000 review cap. "
         "The common-types cohort retains types with at least 100 distinct training buildings. The primary stable-single-building cohort requires "
         "exactly one reported building in each retained record, floor area at least 1,000 ft², and no existing cross-year characteristic review flags. "
         "These are sensitivity cohorts, not a search for the lowest test error. Different cohort errors concern different populations."),
        ("Model portfolio", "Compare a training-only property-type median baseline, ridge regression, random forest, and absolute-error gradient boosting. "
         "All use the same allowed inputs (the median baseline uses type alone). Choose the general family by validation MAE. "
         "Fit specialists only with at least 60 training and 20 validation buildings of that type; choose their family on validation MAE. "
         "Route a type to its specialist only when validation MAE improves at least 5% over the selected general model on the same validation rows. "
         "Otherwise fall back to the general model. These thresholds are practical prespecified settings, not guarantees of adequate power. "
         "Models remain frozen after training. Test results do not alter the routes. Per-type recommendations are provisional because several candidates "
         "are compared on one validation split; repeated nested grouped validation would assess selection stability."),
        ("Property-type performance and fairness", "The main chart compares the selected general model with the best validation-selected specialist "
         "on exactly the same held-out records within each type. It displays types with at least 20 test buildings; the CSV retains every evaluated type "
         "and flags limited test support. Whiskers resample entire buildings, preserving their annual records. "
         "Positive signed error means underprediction, which can raise a type's apparent audit priority. Normalized MAE divides by mean observed EUI "
         "within the type. Macro MAE gives each evaluated type equal weight, including small unstable groups. "
         "A positive-residual rate is a flagging diagnostic, not a false-positive rate: verified audit outcomes are unavailable. "
         "Differences by property type demonstrate performance disparities, not demographic fairness or discrimination. "
         "Absolute-error boosting and median baselines target typical (median) consumption rather than mean consumption; "
         "a right-skewed consumption distribution can therefore produce positive mean signed errors even for useful models."),
        ("Feature importance", "Permutation importance measures the increase in held-out MAE when one input block is shuffled. Location and weather "
         "columns are shuffled jointly within each block, and the portfolio reroutes when property type is shuffled. The CSV also includes selected "
         "specialists with at least 20 test buildings. Whiskers show variation over eight shuffles, not confidence intervals. "
         "Importance is model-specific and associative. Correlation between year, weather, and location can spread or mask importance; these plots "
         "do not establish causal drivers or identify a building's faulty equipment. Target-derived inputs (energy, emissions, ENERGY STAR scores, "
         "weather-normalized EUI) are excluded. Floor area enters as a logarithm."),
        ("Prediction intervals", "A separate calibration sample sets a 90% interval using each calibration building's maximum absolute annual error. "
         f"The portfolio's EUI radius is {primary['portfolio_interval_radius']:.2f}; the observed held-out fraction of buildings with all recorded years "
         f"inside their intervals is {100*primary['test_simultaneous_building_coverage']:.1f}%. "
         "Coverage is marginal across exchangeable buildings, not guaranteed within every property type. Intervals may be wide. "
         "They are prediction intervals, not confidence intervals for avoidable waste."),
        ("Audit screening and manager communication", "The manager example is a held-out building: none of its years trained or selected its prediction model. "
         "The demonstration ranking requires all three years, positive residuals in at least two years, and positive mean annual consumption gap. "
         "Its score is the mean of (actual EUI − predicted EUI) × that year's area; negative annual gaps are retained. "
         "Ratios above five require data review. The example is the highest-scoring supported-type candidate without that ratio flag. "
         "Show the manager the property type, annual actual and expected use, uncertainty, percent difference, and total consumption gap. "
         "A recurring positive gap motivates reviewing meters, reporting boundaries, occupancy, schedules, and equipment. "
         "It neither proves waste nor establishes program eligibility. The new ranking covers held-out primary-cohort buildings only; "
         "the existing full-sample OOF ranking remains a legacy output. A full-population portfolio ranking requires nested building-grouped predictions."),
        ("Scope and next steps", "This experiment estimates generalization to unseen buildings in the recorded 2023–2025 period, not future-year forecasting. "
         "Cross-year quality filters use reported characteristics across the available period; future deployment must compute eligibility using only data available at scoring time. "
         "Interpretation of program year and incomplete latest-year coverage follows the existing project's caveats. "
         "The Boston paper has not been supplied, so these figures are not claimed to reproduce its analysis. "
         "Review its target, sample, validation design, and figure definitions before claiming a replication."),
    ]
    def embed(name):
        data = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
        return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{data}">'
    sections = []
    for title, body in paragraphs:
        sections.append(f"<h2>{html.escape(title)}</h2><p>{html.escape(body)}</p>")
        if title == "Main result":
            sections.append(sel[["strategy", "MAE", "macro_type_MAE", "RMSE", "R2"]].to_html(index=False, float_format=lambda v: f"{v:.3f}"))
            sections.append(embed("07_candidate_models"))
        if title == "Design and filters": sections.append(embed("02_filter_experiments"))
        if title == "Design and filters":
            retention = []
            for s in summaries:
                counts = s["split_counts"]
                retention.append({"Cohort": s["cohort"],
                                  "Total rows": sum(v["rows"] for v in counts.values()),
                                  "Total buildings": sum(v["buildings"] for v in counts.values()),
                                  **{f"{role.title()} buildings": counts[role]["buildings"]
                                     for role in ["train", "validation", "calibration", "test"]}})
            sections.append(pd.DataFrame(retention).to_html(index=False))
        if title == "Model portfolio": sections.append(table.to_html(index=False, float_format=lambda v: f"{v:.2f}", na_rep="Not fitted"))
        if title == "Property-type performance and fairness":
            sections.extend([embed("01_general_vs_specialist"), embed("03_property_type_fairness")])
        if title == "Feature importance": sections.append(embed("04_feature_importance"))
        if title == "Prediction intervals": sections.append(embed("05_actual_vs_predicted"))
        if title == "Audit screening and manager communication" and example:
            sections.append(embed("06_building_manager_example"))
            sections.append(pd.read_csv(OUT / "building_manager_example.csv")[["year", "property_type", "site_eui", "pred_portfolio", "gap_pct", "actual_energy_kbtu", "predicted_energy_kbtu", "above_interval"]].to_html(index=False, float_format=lambda v: f"{v:,.2f}"))
            sections.append(f"<p>Example building {html.escape(str(example['building_id']))}: reported use exceeds expected use in "
                            f"{example['positive_years']} of {example['years']} years, and exceeds the prediction interval in "
                            f"{example['above_interval_years']} years. Average annual consumption gap: "
                            f"{example['mean_annual_excess_kbtu']/1e6:.2f} million kBtu. "
                            "These figures describe a discrepancy to investigate, not promised savings.</p>")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>LA energy audit: property-type results</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#172b3a;max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{margin-top:38px;color:#184f78}img{width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{padding:8px 12px;border-bottom:1px solid #dce4ea;text-align:left}th{background:#eff4f8}p{max-width:1050px}.sub{color:#526575}</style><h1>LA energy audit prioritization</h1><p class="sub">Property-type models, validation-selected portfolio, and building-level explanations · results from the uploaded data</p>' + "\n".join(sections) + '</html>'
    (OUT / "results_report.html").write_text(page)
    md_table = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    for record in table.itertuples(index=False, name=None):
        md_table.append("| " + " | ".join("Not fitted" if pd.isna(v) else
                        (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in record) + " |")
    lines = ["# Updated property-type results", "", paragraphs[0][1], "", "## Recommendations by type", "", *md_table, ""]
    for title, body in paragraphs[1:]: lines += [f"## {title}", "", body, ""]
    lines += ["## Reproduce", "", "```bash", "python scripts/03_clean_buildings.py", "python scripts/05_build_model_table.py", "python scripts/11_property_type_results.py", "pytest -q tests", "```", "", "See `outputs/property_type_results/results_report.html` for the self-contained illustrated report. All new tables and figures are in that folder."]
    (ROOT / "docs/property_type_results.md").write_text("\n".join(lines))


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/model_table.csv", dtype={"building_id": str})
    required = {"needs_data_review", "number_of_buildings"}
    if not required.issubset(df.columns):
        raise ValueError("Model table predates quality checks. Run scripts 03 and 05 first.")
    valid = (df.site_eui.gt(0) & df.site_eui.le(cfg["site_eui_model_max"]) & df.floor_area.gt(0)
             & df.station_id.notna() & df.mean_temp_f.notna() & df.building_id.notna())
    df = df.loc[valid].reset_index(drop=True)
    df["property_type"] = df.property_type.fillna("Unknown")
    df["log_floor_area"] = np.log(df.floor_area)
    df["split"] = assign_splits(df, cfg["random_state"])
    masks = cohort_masks(df, cfg["property_type_analysis"]["common_type_min_train_buildings"])
    membership = df[["building_id", "year", "property_type", "split"]].copy()
    for name, mask in masks.items(): membership[name] = mask
    membership.to_csv(OUT / "split_and_cohort_membership.csv", index=False)
    results, summaries, primary = [], [], None
    for name, mask in masks.items():
        print(f"Fitting cohort {name}: {mask.sum():,} rows", flush=True)
        result = evaluate_cohort(name, df.loc[mask], cfg)
        results.append(result)
        summaries.append(result[4])
        if name == "stable_single_building": primary = result
    test, types, _, _, _, general, specialists, choices = primary
    aggregate = pd.concat([r[2] for r in results], ignore_index=True)
    all_types = pd.concat([r[1] for r in results], ignore_index=True)
    all_selection = pd.concat([r[3] for r in results], ignore_index=True)
    predictions = pd.concat([r[0].assign(cohort=s["cohort"]) for r, s in zip(results, summaries)], ignore_index=True)
    aggregate.to_csv(OUT / "test_metrics.csv", index=False)
    all_types.to_csv(OUT / "per_type_test_metrics.csv", index=False)
    all_selection.to_csv(OUT / "validation_model_selection.csv", index=False)
    predictions.to_csv(OUT / "heldout_predictions.csv", index=False)
    importance_rows = permutation_blocks(test, lambda x: predict(general, x), "stable_single_building", "general", cfg["random_state"])
    importance_rows += permutation_blocks(test, lambda x: route_predict(x, general, specialists, choices), "stable_single_building", "portfolio", cfg["random_state"])
    for typ, route in choices.items():
        group = test[test.property_type.eq(typ)]
        if route != "general" and group.building_id.nunique() >= cfg["property_type_analysis"]["min_report_test_buildings"]:
            importance_rows += permutation_blocks(group, lambda x, m=specialists[typ][route]: predict(m, x), "stable_single_building", f"specialist: {typ}", cfg["random_state"])
    # Single-column importance (blocks such as location and weather split into their columns) and the
    # selected general model scored within each type that has enough test buildings.
    single_columns = {c: [c] for columns in BLOCKS.values() for c in columns}
    importance_rows += permutation_blocks(test, lambda x: predict(general, x), "stable_single_building",
                                          "general_single_columns", cfg["random_state"], blocks=single_columns)
    min_support = cfg["property_type_analysis"]["min_report_test_buildings"]
    within_type = {k: v for k, v in BLOCKS.items() if k != "Property type"}
    for typ, group in test.groupby("property_type"):
        if group.building_id.nunique() >= min_support:
            importance_rows += permutation_blocks(group, lambda x: predict(general, x), "stable_single_building",
                                                  f"general within: {typ}", cfg["random_state"], blocks=within_type)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(OUT / "feature_block_importance.csv", index=False)
    figures(test, types, aggregate, importance)
    example = manager_example(test, types, cfg)
    report = {"settings": cfg["property_type_analysis"], "seed": cfg["random_state"],
              "primary_cohort": "stable_single_building", "cohorts": summaries, "manager_example": example}
    (OUT / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    write_report(summaries, aggregate, types, importance, example, cfg)
    manifest = {
        "analysis_version": "validation_selected_v2",
        "primary_cohort": "stable_single_building", "evaluation": "fixed_heldout_buildings",
        "input_sha256": file_sha256(ROOT / "data/processed/model_table.csv"),
        "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [
            Path(__file__), ROOT / "src/property_models.py", ROOT / "config.yml"]},
        "artifact_sha256": {p.name: file_sha256(p) for p in sorted(OUT.glob("*.csv"))
                            if p.name in {"heldout_predictions.csv", "per_type_test_metrics.csv", "test_metrics.csv",
                                          "validation_model_selection.csv", "split_and_cohort_membership.csv", "feature_block_importance.csv"}
                            or p.name.startswith("calibration_predictions_")},
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved illustrated report: {OUT / 'results_report.html'}", flush=True)


if __name__ == "__main__":
    main()
