"""Extras for the property-type analysis. Run after scripts/11_property_type_results.py.

Reads only the outputs of script 11 (held-out predictions, per-type metrics, importance,
calibration predictions); it trains nothing and does not change any of its numbers.
Writes into outputs/property_type_results/:

  recommendations_by_type.csv / recommendations.md   which model to use for which property type
  interval_coverage_by_type.csv                       does the 90% interval cover each type equally?
  selection_fairness_by_type.csv                      who the audit screen flags, by property type
  figures/08_recommendation_by_type.png               general vs median vs portfolio MAE per type
  figures/09_selection_fairness.png                   relative error and flag rates by type
  figures/10_interval_coverage_by_type.png            constant vs type-scaled interval coverage
  figures/11_actual_vs_predicted_log.png              predicted vs actual (log scale) and residuals
  figures/12_importance_by_type.png                   permutation importance, general and within each type
  figures/building_cards/<id>_audit_card.png          what to show one building manager
and adds the same material to results_report.html.

Why a type-scaled interval: script 11 uses one constant radius for every building. A radius that
covers 90% of buildings overall is far too wide for a 4 kBtu/ft2 self-storage building and too narrow
for a 90 kBtu/ft2 supermarket, so the "above the interval" signal is not equally sensitive across
types. Here the radius is relative to the building's expected use (floor SCALE_FLOOR kBtu/ft2).
"""
from pathlib import Path
import argparse
import base64
import html
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config
from audit_reporting import same_type_context, verify_analysis_files, file_sha256
import json

try:
    from property_models import LABELS
except ImportError:  # keeps the script usable if the module is renamed
    LABELS = {"general": "General model", "property_type_median": "Property-type median",
              "ridge": "Ridge regression", "random_forest": "Random forest",
              "gradient_boosting": "MAE gradient boosting"}

OUT = ROOT / "outputs/property_type_results"
FIG = OUT / "figures"
CARDS = FIG / "building_cards"
PRIMARY = "stable_single_building"
SCALE_FLOOR = 5.0          # kBtu/ft2; keeps the relative interval from collapsing at very low expected EUI
MAX_TYPES_SHOWN = 14
BLUE, ORANGE, GREEN, GRAY, RED = "#2b6cb0", "#dd7a1f", "#23856d", "#8a8f98", "#c0392b"   # palette of script 11
MARKER = "<!-- extras-start -->"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})

SHORT = {"Multifamily Housing": "Multifamily", "Non-Refrigerated Warehouse": "Non-refrig. warehouse",
         "Manufacturing/Industrial Plant": "Manufacturing", "Supermarket/Grocery Store": "Supermarket",
         "Self-Storage Facility": "Self-storage", "Distribution Center": "Distribution ctr",
         "Mixed Use Property": "Mixed use", "Refrigerated Warehouse": "Refrig. warehouse"}


def short(t):
    return SHORT.get(t, t)


def label(route):
    return LABELS.get(route, route)


# ----------------------------------------------------------------------------- statistics helpers
def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n. Returns (p, low, high)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Roundoff at k=0 or k=n can put an endpoint just past p, which
    # produces a tiny negative error-bar length. Keep the exact boundary.
    return p, min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


def conformal_quantile(values, level=.9):
    values = np.asarray(values, float)
    n = len(values)
    if n == 0:
        raise ValueError("Calibration sample is empty")
    rank = int(np.ceil((n + 1) * level))
    return float(np.sort(values)[rank - 1]) if rank <= n else float("inf")


def scaled_radius(cal, pred, floor=SCALE_FLOOR, level=.9):
    """Relative radius q: each calibration building's worst-year |error| / max(expected, floor),
    then the conformal quantile across buildings."""
    pred = np.asarray(pred, float)
    score = pd.Series(np.abs(cal["site_eui"].to_numpy() - pred) / np.maximum(pred, floor), index=cal.index)
    return conformal_quantile(score.groupby(cal["building_id"]).max().to_numpy(), level)


def scaled_interval(pred, q, floor=SCALE_FLOOR):
    pred = np.asarray(pred, float)
    half = q * np.maximum(pred, floor)
    return np.maximum(0, pred - half), pred + half


def building_screen(df, pred_col, above_col, n_years):
    """One row per building. Rule from script 11: all years present, a positive residual in at least two
    years, and a positive mean annual gap (kBtu). 'outside' = above the expected range in >= 2 years."""
    d = df.assign(_res=df["site_eui"] - df[pred_col])
    d["_gap"] = d["_res"] * d["floor_area"]
    g = d.groupby("building_id").agg(
        years=("year", "nunique"), positive_years=("_res", lambda v: int((v > 0).sum())),
        mean_gap_kbtu=("_gap", "mean"), mean_actual=("site_eui", "mean"), mean_expected=(pred_col, "mean"),
        above_range_years=(above_col, "sum"), property_type=("property_type", "first"),
        floor_area=("floor_area", "median"))
    g["three_year"] = g["years"].eq(n_years)
    g["qualifies"] = g["three_year"] & g["positive_years"].ge(2) & g["mean_gap_kbtu"].gt(0)
    g["outside_range"] = g["three_year"] & g["above_range_years"].ge(2)
    g["ratio"] = g["mean_actual"] / g["mean_expected"].where(g["mean_expected"] > 0)
    g["check_data_first"] = g["ratio"].gt(5) | g["mean_expected"].le(0)
    ranked = g[g["qualifies"]].sort_values("mean_gap_kbtu", ascending=False)
    g["rank"] = np.nan
    g.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    top = set(ranked.head(int(np.ceil(len(ranked) * .10))).index)
    g["top_decile"] = g.index.isin(top)
    return g.reset_index()


# ----------------------------------------------------------------------------- loading
def load(cohort):
    verify_analysis_files(OUT)
    held = pd.read_csv(OUT / "heldout_predictions.csv", dtype={"building_id": str})
    held = held[held["cohort"].eq(cohort)].reset_index(drop=True)
    types = pd.read_csv(OUT / "per_type_test_metrics.csv")
    types = types[types["cohort"].eq(cohort)].reset_index(drop=True)
    cal_path = OUT / f"calibration_predictions_{cohort}.csv"
    cal = pd.read_csv(cal_path, dtype={"building_id": str}) if cal_path.exists() else None
    imp_path = OUT / "feature_block_importance.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else None
    return held, types, cal, imp


def shown_types(types, held, k=MAX_TYPES_SHOWN):
    ok = types[~types["limited_test_support"]].sort_values("test_buildings", ascending=False)
    return ok.head(k)["property_type"].tolist()


# ----------------------------------------------------------------------------- recommendations
def recommendations(held, types):
    median_mae = (held["site_eui"] - held["candidate_property_type_median"]).abs().groupby(held["property_type"]).mean()
    rec = types[["property_type", "train_buildings", "validation_buildings", "test_buildings", "limited_test_support",
                 "general_model", "selected_route", "general_MAE", "specialist_MAE", "portfolio_MAE",
                 "portfolio_minus_general_MAE", "paired_diff_ci_low", "paired_diff_ci_high",
                 "portfolio_normalized_MAE_pct", "portfolio_signed_error",
                 "general_MAE_low", "general_MAE_high"]].copy()
    rec["median_MAE"] = rec["property_type"].map(median_mae)

    def advice(r):
        if r.limited_test_support:
            name = label(r.general_model) if r.selected_route == "general" else label(r.selected_route)
            return f"Frozen validation route: {name}; insufficient test support for an individual claim"
        if r.selected_route == "general":
            return f"General model: {label(r.general_model)}"
        better = r.paired_diff_ci_high < 0
        return (f"Specialist: {label(r.selected_route)}" +
                (" (test error lower, interval excludes 0)" if better else " (chosen on validation; test difference not distinguishable from 0)"))

    rec["recommendation"] = rec.apply(advice, axis=1)
    rec["specialist_significantly_better"] = (rec["selected_route"].ne("general") & rec["paired_diff_ci_high"].lt(0))
    return rec.sort_values("test_buildings", ascending=False).reset_index(drop=True)


def markdown_table(df):
    """Plain markdown table (avoids the optional `tabulate` dependency of DataFrame.to_markdown)."""
    cell = lambda v: "" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(lines)


def recommendation_text(rec):
    ok = rec[~rec["limited_test_support"]]
    spec = ok[ok["selected_route"].ne("general")]
    gen = ok[ok["selected_route"].eq("general")]
    general_name = label(rec["general_model"].iloc[0])
    parts = []
    if len(spec):
        items = "; ".join(f"{short(r.property_type)} → {label(r.selected_route)}"
                          + (" (test CI excludes 0)" if r.specialist_significantly_better else " (not distinguishable from general on test)")
                          for r in spec.itertuples())
        parts.append(f"Use a specialist for: {items}.")
    else:
        parts.append("No adequately evaluated property type has a validation-selected specialist route.")
    parts.append(f"Use the general model ({general_name}) for: " + (", ".join(short(t) for t in gen["property_type"]) or "none") + ".")
    n_small = int(rec["limited_test_support"].sum())
    parts.append(f"The remaining {n_small} property types have too few test buildings for reliable individual conclusions; "
                 "their frozen validation routes and descriptive errors remain in the CSV. Test support does not change the selected route.")
    return " ".join(parts)


def fig_recommendation(rec, held):
    d = rec[~rec["limited_test_support"]].head(MAX_TYPES_SHOWN).copy()
    if d.empty:
        return
    y = np.arange(len(d))
    h = .27
    fig, ax = plt.subplots(figsize=(12, max(5, .62 * len(d) + 1.6)))
    ax.barh(y - h, d["median_MAE"], height=h, color=GRAY, label="Property-type median baseline")
    ax.barh(y, d["general_MAE"], height=h, color=BLUE, label=f"Selected general model: {label(d['general_model'].iloc[0])}")
    ax.barh(y + h, d["portfolio_MAE"], height=h, color=GREEN, label="Selected portfolio")
    ax.errorbar(d["general_MAE"], y, xerr=[np.maximum(0, d["general_MAE"] - d["general_MAE_low"]),
                                           np.maximum(0, d["general_MAE_high"] - d["general_MAE"])], fmt="none", ecolor="#333", capsize=2)
    xmax = float(d[["median_MAE", "general_MAE", "portfolio_MAE", "general_MAE_high"]].max().max())
    for yi, r in zip(y, d.itertuples()):
        routed = r.selected_route != "general"
        ax.text(xmax * 1.06, yi + h, ("Specialist: " + label(r.selected_route)) if routed else "General model",
                va="center", fontsize=9, color=ORANGE if routed else "#555", fontweight="bold" if routed else "normal")
    ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(d["property_type"], d["test_buildings"])])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.55)
    ax.set(xlabel="Held-out MAE (kBtu/ft²/year); lower is better",
           title="Which model to use for which property type · stable single-building cohort")
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.1), ncol=3, fontsize=8, frameon=False)
    fig.text(.01, -.06, "n = test buildings. Whiskers: 95% building bootstrap of the general model's MAE. Orange label: validation routed this type to a specialist. "
             "Types with fewer test buildings than the reporting threshold are omitted.", fontsize=8)
    fig.savefig(FIG / "08_recommendation_by_type.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- intervals
def add_intervals(held, cal):
    """Add scaled-interval columns to the held-out table. Returns (held, q) or (held, None)."""
    if cal is None:
        return held, None
    q = scaled_radius(cal, cal["pred_portfolio"].to_numpy())
    lo, hi = scaled_interval(held["pred_portfolio"].to_numpy(), q)
    held = held.assign(scaled_low=lo, scaled_high=hi, above_scaled=held["site_eui"] > hi)
    return held, q


def coverage_table(held, types_to_show):
    rows = []
    for t in types_to_show + ["ALL"]:
        g = held if t == "ALL" else held[held["property_type"].eq(t)]
        const_in = g["site_eui"].between(g["interval_low"], g["interval_high"]).groupby(g["building_id"]).all()
        row = {"property_type": t, "test_buildings": g["building_id"].nunique(),
               "constant_coverage": const_in.mean(), "constant_median_width": float((g["interval_high"] - g["interval_low"]).median())}
        _, row["constant_ci_low"], row["constant_ci_high"] = wilson(int(const_in.sum()), len(const_in))
        if "scaled_low" in g:
            sc_in = g["site_eui"].between(g["scaled_low"], g["scaled_high"]).groupby(g["building_id"]).all()
            row.update({"scaled_coverage": sc_in.mean(), "scaled_median_width": float((g["scaled_high"] - g["scaled_low"]).median())})
            _, row["scaled_ci_low"], row["scaled_ci_high"] = wilson(int(sc_in.sum()), len(sc_in))
        rows.append(row)
    return pd.DataFrame(rows)


def fig_coverage(cov):
    d = cov[cov["property_type"].ne("ALL")].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    h = .36
    fig, ax = plt.subplots(figsize=(11, max(4.6, .5 * len(d) + 1.6)))
    for off, pre, col, name in [(h / 2, "constant", BLUE, "Constant radius (script 11)"), (-h / 2, "scaled", GREEN, "Type-scaled radius")]:
        if f"{pre}_coverage" not in d:
            continue
        ax.barh(y + off, d[f"{pre}_coverage"] * 100, height=h, color=col, label=name,
                xerr=[np.maximum(0, (d[f"{pre}_coverage"] - d[f"{pre}_ci_low"]) * 100),
                      np.maximum(0, (d[f"{pre}_ci_high"] - d[f"{pre}_coverage"]) * 100)],
                error_kw={"ecolor": "#333", "capsize": 2, "lw": 1})
    ax.axvline(90, color="#555", ls="--", lw=1)
    ax.text(90.6, len(d) - .45, "90% target", fontsize=8, color="#555")
    ax.set_yticks(y, [f"{t}  (n={n})" for t, n in zip(d["property_type"], d["test_buildings"])])
    ax.set_xlim(0, 108)
    ax.set(xlabel="Held-out buildings with every year inside the interval (%)",
           title="Does the 90% prediction interval cover every property type equally?")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(.01, -.04, "Whiskers: 95% Wilson interval. Coverage is not uniform across types under either interval, so the 'above the expected range' "
             "signal is not equally sensitive for every property type. Scaled: radius proportional to each building's expected use.", fontsize=8)
    fig.savefig(FIG / "10_interval_coverage_by_type.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- fairness by type
def selection_table(screen, types_to_show):
    base = screen[screen["three_year"]]
    rows = []
    for t in types_to_show + ["ALL"]:
        g = base if t == "ALL" else base[base["property_type"].eq(t)]
        n = len(g)
        row = {"property_type": t, "three_year_buildings": n}
        for name, col in [("meets_screen", "qualifies"), ("outside_range", "outside_range"), ("top_decile", "top_decile")]:
            k = int(g[col].sum())
            p, lo, hi = wilson(k, n)
            row.update({f"{name}_count": k, f"{name}_rate": p, f"{name}_ci_low": lo, f"{name}_ci_high": hi})
        rows.append(row)
    out = pd.DataFrame(rows)
    overall = out.loc[out["property_type"].eq("ALL")].iloc[0]
    for name in ["meets_screen", "outside_range", "top_decile"]:
        out[f"{name}_relative_to_all"] = out[f"{name}_rate"] / overall[f"{name}_rate"] if overall[f"{name}_rate"] > 0 else np.nan
    return out


def fig_fairness(held, types, sel, types_to_show):
    d = types.set_index("property_type").loc[types_to_show]
    gen_norm = {t: 100 * (g["site_eui"] - g["pred_general"]).abs().mean() / g["site_eui"].mean()
                for t, g in held.groupby("property_type") if t in types_to_show}
    sel_d = sel.set_index("property_type")
    y = np.arange(len(types_to_show))
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4.8, .5 * len(y) + 1.8)), sharey=True)
    h = .36
    ax = axes[0]
    ax.barh(y - h / 2, [gen_norm[t] for t in types_to_show], height=h, color=BLUE, label="Selected general model")
    ax.barh(y + h / 2, [d.loc[t, "portfolio_normalized_MAE_pct"] for t in types_to_show], height=h, color=GREEN, label="Selected portfolio")
    ax.set(xlabel="100 × MAE / mean observed EUI (%)", title="Error relative to each type's mean EUI")
    ax.legend(loc="lower right", fontsize=8)
    overall = sel_d.loc["ALL"]
    for ax, name, title in [(axes[1], "meets_screen", "Share meeting the audit screen"),
                            (axes[2], "outside_range", "Share above the expected range in 2+ years")]:
        r = sel_d.loc[types_to_show]
        vals = r[f"{name}_rate"] * 100
        ax.barh(y, vals, color=[ORANGE if v > overall[f"{name}_rate"] * 100 else BLUE for v in vals],
                xerr=[np.maximum(0, (r[f"{name}_rate"] - r[f"{name}_ci_low"]) * 100),
                      np.maximum(0, (r[f"{name}_ci_high"] - r[f"{name}_rate"]) * 100)],
                error_kw={"ecolor": "#333", "capsize": 2, "lw": 1})
        ax.axvline(overall[f"{name}_rate"] * 100, color="#555", ls="--", lw=1)
        ax.set(xlabel="% of three-year held-out buildings", title=title)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y, [f"{t}  (n={int(sel_d.loc[t, 'three_year_buildings'])})" for t in types_to_show])
    axes[0].invert_yaxis()
    fig.suptitle("Performance and screening rates by property type · stable single-building cohort", y=1.02)
    fig.text(.01, -.03, "Dashed line: all types. Orange: above the all-type rate; blue: at or below it. Whiskers: 95% Wilson interval; small types are noisy. "
             "Differences are between building uses, not a demographic fairness measure.", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "09_selection_fairness.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- Boston-style predicted vs actual
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def fig_actual_vs_predicted(held, types_to_show):
    y = held["site_eui"].to_numpy()
    gen, port = held["pred_general"].to_numpy(), held["pred_portfolio"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))

    def panel(ax, actual, pred, title, color):
        cap = np.quantile(np.r_[actual, pred], .99)
        ax.scatter(actual, pred, s=9, alpha=.22, color=color)
        ax.plot([0, cap], [0, cap], "--", color="#555")
        ax.set(xlim=(0, cap), ylim=(0, cap), xlabel="Actual Site EUI (kBtu/ft²/year)", ylabel="Predicted Site EUI", title=title)
        ax.text(.04, .95, f"n = {len(actual):,} building-years\nMAE = {np.abs(actual - pred).mean():.1f}\nR² = {r2(actual, pred):.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5)

    panel(axes[0, 0], y, gen, "Held-out actual versus predicted · selected general model", BLUE)
    panel(axes[0, 1], y, port, "Held-out actual versus predicted · selected portfolio", GREEN)
    res_g, res_p = y - gen, y - port
    bins = np.linspace(-120, 120, 80)
    ax = axes[0, 2]
    ax.hist(res_g, bins=bins, color=BLUE, alpha=.7, label=f"General (median {np.median(res_g):+.1f})")
    ax.hist(res_p, bins=bins, color=GREEN, alpha=.6, label=f"Portfolio (median {np.median(res_p):+.1f})")
    ax.axvline(0, color="#555", lw=.8)
    ax.set(xlabel="Residual = actual − predicted (kBtu/ft²/year)", ylabel="Building-years", title="Residual distribution")
    ax.legend(fontsize=8)
    for ax, t in zip(axes[1], (types_to_show + [None] * 3)[:3]):
        if t is None:
            ax.axis("off")
            continue
        m = held["property_type"].eq(t).to_numpy()
        panel(ax, y[m], port[m], f"{t} · portfolio", GREEN)
    fig.text(.01, -.01, "Axes end at each panel's 99th percentile for readability; metrics include all held-out rows. Residual histogram clipped at ±120.", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "11_actual_vs_predicted_log.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- importance
NICE = {"floor_area": "Floor area", "log_floor_area": "Floor area", "building_age": "Building age", "latitude": "Latitude",
        "longitude": "Longitude", "year": "Reporting year", "mean_temp_f": "Mean temp", "total_prcp_mm": "Precipitation",
        "hdd65": "Heating degree days", "cdd65": "Cooling degree days", "hot_days_90f": "Days above 90°F",
        "property_type": "Property type"}


def fig_importance(imp, cohort):
    if imp is None:
        return False
    imp = imp[imp["cohort"].eq(cohort)]
    single = imp[imp["scope"].eq("general_single_columns")].copy()
    within = imp[imp["scope"].str.startswith("general within: ")].copy()
    blocks = imp[imp["scope"].eq("general")]
    if single.empty and within.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    if not single.empty:
        s = single.sort_values("MAE_increase")
        ax.barh(s["feature_block"].map(lambda c: NICE.get(c, c)), s["MAE_increase"], xerr=s["repeat_std"], capsize=3, color=BLUE)
        ax.axvline(0, color="#555", lw=.8)
        ax.set(xlabel="Test MAE increase after permutation (kBtu/ft²)", title="Selected general model · single columns")
        notes = []
        pt = blocks[blocks["feature_block"].eq("Property type")]
        wb = blocks[blocks["feature_block"].str.contains("weather", case=False)]
        if len(pt):
            notes.append(f"Property type (block): +{pt['MAE_increase'].iloc[0]:.2f}")
        if len(wb):
            notes.append(f"All weather together: +{wb['MAE_increase'].iloc[0]:.2f}")
        if notes:
            ax.text(.97, .04, "\n".join(notes), transform=ax.transAxes, ha="right", fontsize=8.5, bbox=dict(boxstyle="round", fc="#eff4f8", ec="none"))
    else:
        ax.axis("off")
    ax = axes[1]
    if not within.empty:
        within["type"] = within["scope"].str.replace("general within: ", "", regex=False)
        within["pct"] = 100 * within["MAE_increase"] / within["baseline_MAE"]
        mat = within.pivot_table(index="type", columns="feature_block", values="pct")
        order = within.drop_duplicates("type").sort_values("test_buildings", ascending=False)["type"].tolist()
        mat = mat.loc[order]
        im = ax.imshow(mat.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(float(np.nanmax(mat.to_numpy())), 1.0))
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=30, ha="right")
        ax.set_yticks(range(mat.shape[0]), list(mat.index))
        top = np.nanmax(mat.to_numpy())
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8.5, color="white" if mat.iloc[i, j] > .6 * top else "#222")
        ax.set_title("Within each type · MAE increase as % of that type's MAE")
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.colorbar(im, ax=ax, fraction=.04, pad=.02).outline.set_visible(False)
    else:
        ax.axis("off")
    fig.suptitle("Feature permutation importance · larger means more predictive information", y=1.02)
    fig.text(.01, -.02, "Whiskers: variation over repeated shuffles, not confidence intervals. Importance is model-specific and associative, not causal.", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "12_importance_by_type.png")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- building cards
def building_card(held, screen, building_id, q_text):
    rows = held[held["building_id"].eq(building_id)].sort_values("year")
    if rows.empty:
        raise SystemExit(f"building_id {building_id} is not among the held-out buildings of the primary cohort")
    b = screen[screen["building_id"].eq(building_id)].iloc[0]
    context = same_type_context(screen, building_id)
    n_rank = int(screen["qualifies"].sum())
    route = rows["selected_route"].iloc[0]
    model_name = label(rows["general_model"].iloc[0]) if route == "general" else label(route)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.15, .8, 1.05]})
    years = rows["year"].astype(int).tolist()
    x = np.arange(len(years))
    pred = rows["pred_portfolio"].to_numpy()
    low, high = rows["band_low"].to_numpy(), rows["band_high"].to_numpy()
    ax = axes[0]
    ax.bar(x - .18, rows["site_eui"], .35, color=ORANGE, label="Reported")
    ax.bar(x + .18, pred, .35, color=BLUE, label="Expected")
    ax.errorbar(x + .18, pred, yerr=[np.maximum(0, pred - low), np.maximum(0, high - pred)], fmt="none", color="#333", capsize=4)
    top = max(rows["site_eui"].max(), high.max())
    for xi, (s, hi) in enumerate(zip(rows["site_eui"], rows["above_band"])):
        ax.text(xi - .18, s + top * .015, f"{s:.0f}", ha="center", fontsize=9, fontweight="bold")
        if hi:
            ax.text(xi - .18, s + top * .075, "above range", ha="center", fontsize=8, color=RED)
    for xi, e in enumerate(pred):
        ax.text(xi + .18, top * .012, f"{e:.0f}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x, years)
    ax.set_ylim(0, top * 1.16)
    ax.set(ylabel="Site EUI (kBtu/ft²/year)", title="Reported versus expected use")
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1]
    gap = rows["eui_residual"].to_numpy() * rows["floor_area"].to_numpy() / 1e6
    ax.bar(x, gap, color=GREEN)
    ax.axhline(0, color="#555", lw=.8)
    ax.set_xticks(x, years)
    ax.set(ylabel="Reported minus expected (million kBtu/year)", title="Consumption gap to investigate")
    t = axes[2]
    t.axis("off")
    zip_txt = f", ZIP {int(rows['postal_code'].iloc[0])}" if "postal_code" in rows and pd.notna(rows["postal_code"].iloc[0]) else ""
    t.text(0, 1.0, f"{b['property_type']}, {b['floor_area']:,.0f} ft²{zip_txt}", fontsize=10.5, fontweight="bold", va="top")
    if b["qualifies"]:
        head, col = "Meets the audit screen", RED
        lines = [f"Reported use was above expected in {int(b['positive_years'])} of {int(b['years'])} years, and above the expected range in {int(b['above_range_years'])}.",
                 f"Average: {b['mean_actual']:.0f} reported vs. {b['mean_expected']:.0f} expected kBtu/ft² ({100 * (b['mean_actual'] / b['mean_expected'] - 1):+.0f}%).",
                 f"That gap is about {b['mean_gap_kbtu'] / 1000:,.0f} MMBtu per year.",
                 f"Rank {int(b['rank']):,} of {n_rank:,} held-out buildings that meet the screen (ranked among held-out buildings only)."]
        if b["check_data_first"]:
            lines.append("Reported use is more than 5x expected. Check the utility data and the reporting boundary first.")
    else:
        head, col = "Does not meet the audit screen", GREEN
        why = []
        if b["years"] < 3:
            why.append(f"only {int(b['years'])} of 3 years have usable data")
        if b["positive_years"] < 2:
            why.append(f"reported use was above expected in only {int(b['positive_years'])} of {int(b['years'])} years")
        if b["mean_gap_kbtu"] <= 0:
            why.append("average use is at or below expected")
        lines = [f"Reason: {'; '.join(why)}.", f"Average: {b['mean_actual']:.0f} reported vs. {b['mean_expected']:.0f} expected kBtu/ft²."]
    t.text(0, .91, head, fontsize=13, fontweight="bold", color=col, va="top")
    lines.append(f"Comparison context: {context['same_type_three_year_buildings']:,} three-year {b['property_type']} buildings in the held-out sample.")
    ypos = .81
    for ln in lines:
        wrapped = textwrap.fill(ln, 50)
        t.text(0, ypos, wrapped, fontsize=9.2, va="top")
        ypos -= .052 * (wrapped.count("\n") + 1) + .022
    fig.suptitle(f"Building {building_id} · {b['property_type']}", y=1.02)
    note = (f"Expected use: {model_name} for this property type, fitted on other buildings only. Whiskers: {q_text}. "
            "A gap is a reason to look, not proof of waste: occupancy hours, process loads or reporting errors can explain it.")
    fig.text(.01, -.04, textwrap.fill(note, 190), fontsize=8)
    fig.tight_layout()
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{building_id}_audit_card.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def pick_cards(screen, types_to_show, k=5):
    ok = screen[screen["qualifies"] & ~screen["check_data_first"] & screen["floor_area"].ge(20000)]
    picks = []
    for t in types_to_show[:4]:
        s = ok[ok["property_type"].eq(t)].sort_values("rank")
        if len(s):
            picks.append(s.iloc[min(1, len(s) - 1)]["building_id"])
    neg = screen[screen["three_year"] & ~screen["qualifies"] & screen["mean_gap_kbtu"].lt(0) & screen["property_type"].eq(types_to_show[0])]
    if len(neg):
        picks.append(neg.iloc[len(neg) // 2]["building_id"])
    flagged = screen[screen["qualifies"] & screen["check_data_first"]].sort_values("rank")
    if len(flagged):
        picks.append(flagged.iloc[0]["building_id"])
    return list(dict.fromkeys(picks))[:k]


# ----------------------------------------------------------------------------- report integration
def embed(name):
    p = FIG / f"{name}.png"
    if not p.exists():
        return ""
    return f'<img alt="{html.escape(name)}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def inject_into_report(rec, takeaway, cov, sel, card_ids, q, have_importance):
    path = OUT / "results_report.html"
    if not path.exists():
        return
    page = path.read_text()
    page = re.sub(re.escape(MARKER) + r".*?(?=</html>)", "", page, flags=re.S)
    table = rec[~rec["limited_test_support"]][["property_type", "test_buildings", "general_MAE", "median_MAE", "specialist_MAE", "portfolio_MAE", "recommendation"]]
    parts = [MARKER,
             "<h2>Recommendation by property type</h2>", f"<p>{html.escape(takeaway)}</p>",
             table.to_html(index=False, float_format=lambda v: f"{v:.2f}", na_rep="Not fitted"), embed("08_recommendation_by_type"),
             "<h2>Predicted versus actual</h2><p>Held-out building-years on linear axes limited to each panel's 99th percentile for readability; metrics use all records. The diagonal is a perfect prediction. "
             "Residuals are actual minus predicted EUI.</p>", embed("11_actual_vs_predicted_log"),
             "<h2>Fairness across property types</h2><p>Relative error and screening rates by type. Differences show uneven model performance "
             "and uneven audit flagging across building uses; they say nothing about demographic fairness. Rates carry 95% Wilson intervals "
             "and small types are noisy.</p>", embed("09_selection_fairness")]
    if cov is not None and "scaled_coverage" in cov:
        a = cov[cov["property_type"].eq("ALL")].iloc[0]
        parts += ["<h2>Interval coverage by property type</h2>",
                  f"<p>The constant-radius interval covers {100 * a['constant_coverage']:.1f}% of held-out buildings overall and a type-scaled "
                  f"alternative {100 * a['scaled_coverage']:.1f}%. Neither guarantees coverage within a property type; the figure shows how uneven it is, "
                  "which means the 'above the expected range' signal is more sensitive for some building uses than others.</p>", embed("10_interval_coverage_by_type")]
    if have_importance:
        parts += ["<h2>Feature importance by property type</h2><p>Permutation importance of single columns for the selected general model, "
                  "and the same model scored within each type with enough test buildings. Importance is model-specific and associative.</p>",
                  embed("12_importance_by_type")]
    if card_ids:
        parts += ["<h2>What a building manager would see</h2><p>Example cards for held-out buildings. The first meets the screen; "
                  "others show a building that does not and one that needs a data check first.</p>"] + \
                 [f'<img alt="audit card" src="data:image/png;base64,{base64.b64encode((CARDS / f"{c}_audit_card.png").read_bytes()).decode()}">' for c in card_ids]
    page = page.replace("</html>", "\n".join(parts) + "\n</html>")
    path.write_text(page)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=PRIMARY)
    ap.add_argument("--building-id", action="append", help="draw a card for this held-out building (repeatable)")
    ap.add_argument("--interval", choices=["constant", "scaled"], default="constant",
                    help="interval used for cards and the screen: script 11's constant radius (default) or the type-scaled radius")
    ap.add_argument("--no-report", action="store_true", help="do not modify results_report.html")
    args = ap.parse_args()
    cfg = load_config()
    n_years = len(cfg["years"])
    FIG.mkdir(parents=True, exist_ok=True)
    held, types, cal, imp = load(args.cohort)
    held, q = add_intervals(held, cal)
    if q is None:
        raise SystemExit("calibration_predictions_<cohort>.csv is missing: rerun scripts/11_property_type_results.py (patched version).")
    if args.interval == "scaled":
        held = held.assign(band_low=held["scaled_low"], band_high=held["scaled_high"], above_band=held["above_scaled"])
        band_text = (f"90% interval calibrated on separate buildings (radius {100 * q:.0f}% of expected use, at least {SCALE_FLOOR:g} kBtu/ft² scale)")
    else:
        held = held.assign(band_low=held["interval_low"], band_high=held["interval_high"], above_band=held["above_interval"])
        radius = float((held["interval_high"] - held["pred_portfolio"]).median())
        band_text = f"90% building-level prediction interval calibrated on separate buildings (radius ±{radius:.0f} kBtu/ft²)"
    shown = shown_types(types, held)
    print(f"{len(held):,} held-out rows, {held['building_id'].nunique():,} buildings; {len(shown)} types with enough test buildings")

    rec = recommendations(held, types)
    rec.to_csv(OUT / "recommendations_by_type.csv", index=False)
    takeaway = recommendation_text(rec)
    (OUT / "recommendations.md").write_text("# Which model to use\n\n" + takeaway + "\n\n" +
        rec[~rec["limited_test_support"]][["property_type", "test_buildings", "general_MAE", "median_MAE", "portfolio_MAE", "recommendation"]]
        .pipe(markdown_table) + "\n")
    fig_recommendation(rec, held)
    print("Takeaway:", takeaway)

    cov = coverage_table(held, shown)
    cov.to_csv(OUT / "interval_coverage_by_type.csv", index=False)
    fig_coverage(cov)

    screen = building_screen(held, "pred_portfolio", "above_band", n_years)
    sel = selection_table(screen, shown)
    sel.to_csv(OUT / "selection_fairness_by_type.csv", index=False)
    screen.to_csv(OUT / "heldout_screen.csv", index=False)
    fig_fairness(held, types, sel, shown)
    fig_actual_vs_predicted(held, shown)
    have_imp = fig_importance(imp, args.cohort)

    q_text = band_text
    ids = [str(i) for i in (args.building_id or pick_cards(screen, shown))]
    for bid in ids:
        building_card(held, screen, bid, q_text)
    if not args.no_report:
        inject_into_report(rec, takeaway, cov, sel, ids, q, have_imp)
    card_metadata = []
    for bid in ids:
        context = same_type_context(screen, bid)
        card_metadata.append({"building_id": bid, **context, "cohort": args.cohort,
                              "interval_method": args.interval,
                              "interval_definition": band_text})
    pd.DataFrame(card_metadata).to_csv(OUT / "building_card_metadata.csv", index=False)
    artifacts = ["recommendations_by_type.csv", "interval_coverage_by_type.csv", "selection_fairness_by_type.csv",
                 "heldout_screen.csv", "building_card_metadata.csv",
                 "figures/08_recommendation_by_type.png", "figures/09_selection_fairness.png",
                 "figures/10_interval_coverage_by_type.png", "figures/11_actual_vs_predicted_log.png"]
    if have_imp:
        artifacts.append("figures/12_importance_by_type.png")
    artifacts += [f"figures/building_cards/{bid}_audit_card.png" for bid in ids]
    (OUT / "extras_manifest.json").write_text(json.dumps({
        "analysis_manifest_sha256": file_sha256(OUT / "run_manifest.json"),
        "cohort": args.cohort, "interval_method": args.interval, "card_ids": ids,
        "interval_definition": band_text,
        "artifact_sha256": {name: file_sha256(OUT / name) for name in artifacts}}, indent=2))
    a = cov[cov["property_type"].eq("ALL")].iloc[0]
    print(f"Coverage overall: constant {a['constant_coverage']:.3f}, scaled {a['scaled_coverage']:.3f}; cards: {ids}")


if __name__ == "__main__":
    main()
