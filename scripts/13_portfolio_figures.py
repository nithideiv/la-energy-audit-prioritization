"""Figures and the fairness table for the per-type / portfolio analysis.

Run after 11_per_type_models.py and 12_feature_importance.py. Trains nothing.

Writes (outputs/figures/):
  report_fig6_per_type_vs_general.png   bars per property type: per-type model vs general model (MAE and R2)
  report_fig7_grouping_comparison.png   how each way of splitting the data compares overall
  report_fig8_portfolio.png             recommended model per property type (the takeaway figure)
  report_fig9_pred_vs_actual.png        predicted vs actual Site EUI and residual spread
  report_fig10_feature_importance.png   permutation importance, general model and within large types
  report_fig11_fairness.png             error, bias and audit-selection rate by property type
and outputs/fairness_by_type.csv.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
BLUE, ORANGE, GRAY, RED, GREEN = "#2b6cb0", "#dd7a1f", "#8a8f98", "#c0392b", "#4a8a5b"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 200,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#333", "ytick.color": "#333",
})

SHORT = {"Multifamily Housing": "Multifamily", "Non-Refrigerated Warehouse": "Non-refrig. warehouse",
         "Manufacturing/Industrial Plant": "Manufacturing", "Supermarket/Grocery Store": "Supermarket",
         "Self-Storage Facility": "Self-storage", "Mixed Use Property": "Mixed use",
         "Refrigerated Warehouse": "Refrig. warehouse", "Distribution Center": "Distribution ctr",
         "Hospital (General Medical & Surgical)": "Hospital"}
short = lambda t: SHORT.get(t, t)


def family_color(model):
    if model == "median":
        return GRAY
    return ORANGE if "_rf" in model else BLUE


def reference_models():
    """Names of the general and per-type reference models chosen in 11_per_type_models.py."""
    import json
    s = json.loads((OUT / "portfolio_summary.json").read_text(encoding="utf-8"))
    return s["reference_general_model"], s["reference_per_type_model"]


def load():
    cand = pd.read_csv(OUT / "per_type_oof_predictions.csv", dtype={"building_id": str})
    base = pd.read_csv(OUT / "oof_predictions.csv", dtype={"building_id": str})
    assert (cand["building_id"].to_numpy() == base["building_id"].to_numpy()).all()
    for c in ["floor_area", "address", "emissions", "occupancy"]:
        cand[c] = base[c].to_numpy()
    return (cand, pd.read_csv(OUT / "per_type_metrics.csv"), pd.read_csv(OUT / "model_portfolio.csv"),
            pd.read_csv(OUT / "grouping_comparison.csv"))


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def fig6(cand, per_type, grouping):
    gen_name, typ_name = reference_models()
    names = grouping.set_index("model")["label"]
    types = (cand["property_type"].value_counts().head(12).index).tolist()
    rows = []
    for t in types:
        s = per_type[per_type["property_type"] == t].set_index("model")
        rows.append({"label": f"{short(t)}\n(n={int(s['n_buildings'].iloc[0])})",
                     "gen_mae": s.loc[gen_name, "MAE"], "typ_mae": s.loc[typ_name, "MAE"], "med_mae": s.loc["median", "MAE"],
                     "gen_r2": s.loc[gen_name, "R2"], "typ_r2": s.loc[typ_name, "R2"], "med_r2": s.loc["median", "R2"]})
    y, g, p, m = (cand["site_eui"].to_numpy(), cand[gen_name].to_numpy(), cand[typ_name].to_numpy(), cand["median"].to_numpy())
    rows.append({"label": f"All types\n(n={cand['building_id'].nunique():,})",
                 "gen_mae": np.abs(y - g).mean(), "typ_mae": np.abs(y - p).mean(), "med_mae": np.abs(y - m).mean(),
                 "gen_r2": r2(y, g), "typ_r2": r2(y, p), "med_r2": r2(y, m)})
    d = pd.DataFrame(rows)
    x = np.arange(len(d))
    w = .38
    fig, ax = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True)
    for a, (gc, tc, name) in zip(ax, [("gen_mae", "typ_mae", "MAE (kBtu/ft²), lower is better"),
                                      ("gen_r2", "typ_r2", "R² within type, higher is better")]):
        b1 = a.bar(x - w / 2, d[gc], w, color=GRAY, label=f"General model: {names[gen_name]}")
        b2 = a.bar(x + w / 2, d[tc], w, color=ORANGE, label=f"Per-type model: {names[typ_name]}")
        mc = gc.replace("gen", "med")
        a.scatter(x, d[mc], marker="_", s=420, color="#111", zorder=4, linewidths=1.6)
        for bars in (b1, b2):
            for r in bars:
                v = r.get_height()
                a.text(r.get_x() + r.get_width() / 2, v + (.01 * np.nanmax(np.abs(d[[gc, tc]].to_numpy())) if v >= 0 else
                       -.01 * np.nanmax(np.abs(d[[gc, tc]].to_numpy()))),
                       f"{v:.1f}" if "mae" in gc else f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        a.set_ylabel(name)
        a.axhline(0, color="#999", lw=.6)
    ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + [Line2D([], [], marker="_", ls="", color="#111", ms=12, mew=1.6,
                 label="Property-type median (no model)")], frameon=False, fontsize=8, loc="upper right")
    ax[0].set_title("Per-type models vs. the general model, grouped out-of-fold (same folds)", loc="left")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([l.replace("Non-refrig. warehouse", "Non-refrig.\nwarehouse").replace("Distribution ctr", "Distrib.\ncenter") for l in d["label"]], fontsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG / "report_fig6_per_type_vs_general.png")
    plt.close()
    d.to_csv(OUT / "per_type_vs_general_bars.csv", index=False)


def fig7(grouping):
    d = grouping.sort_values("MAE", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(d))[::-1]
    for a, col, title in [(ax[0], "MAE", "MAE (kBtu/ft²), lower is better"), (ax[1], "R2", "R², higher is better")]:
        bars = a.barh(y, d[col], color=[family_color(m) for m in d["model"]], height=.7)
        for bar, m in zip(bars, d["model"]):
            if m.endswith("_log"):
                bar.set_hatch("////")
                bar.set_edgecolor("white")
        for yi, v in zip(y, d[col]):
            a.text(v + .01 * d[col].abs().max(), yi, f"{v:.2f}", va="center", fontsize=8)
        a.set_xlabel(title)
        a.axvline(0, color="#999", lw=.6)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(d["label"], fontsize=8)
    ax[0].set_xlim(0, d["MAE"].max() * 1.15)
    ax[0].set_title("All ways of splitting the data, overall", loc="left")
    ax[1].legend(handles=[Patch(color=GRAY, label="Median baseline"), Patch(color=BLUE, label="Linear / ridge"),
                          Patch(color=ORANGE, label="Random forest"),
                          Patch(facecolor="#bbb", hatch="////", edgecolor="white", label="Hatched = log-target version")],
               frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, .62))
    plt.tight_layout()
    plt.savefig(FIG / "report_fig7_grouping_comparison.png")
    plt.close()


def fig8(portfolio):
    d = portfolio.copy()
    d["is_other"] = d["segment"].eq("All other types")
    d = d.sort_values(["is_other", "n_rows"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(d))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(10, 6.8))
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.mae_recommended, r.mae_general_ref], [yi, yi], color="#ccc", lw=1.4, zorder=1)
    ax.scatter(d["mae_general_ref"], y, s=34, color=GRAY, zorder=2, label="General model")
    ax.scatter(d["mae_type_median"], y, s=34, facecolors="none", edgecolors="#333", marker="D", zorder=2,
               label="Property-type median")
    ax.scatter(d["mae_recommended"], y, s=64, color=[family_color(m) for m in d["recommended_model"]], zorder=3,
               edgecolors=["#000" if s else "none" for s in d["significantly_better_than_general_ref"]], linewidths=1.2)
    xmax = d[["mae_general_ref", "mae_type_median", "mae_recommended"]].to_numpy().max()
    for yi, r in zip(y, d.itertuples()):
        ax.text(xmax * 1.04, yi, f"{r.recommended_label}", va="center", fontsize=7.8,
                color=family_color(r.recommended_model), fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(s)}  (n={int(n):,})" for s, n in zip(d["segment"], d["n_buildings"])], fontsize=8)
    ax.set_xlim(0, xmax * 1.55)
    ax.set_xlabel("Grouped out-of-fold MAE (kBtu/ft²)")
    ax.set_title("Which model to use for which property type", loc="left")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=GRAY, label="General model (best general ML model)"),
        Line2D([], [], marker="D", ls="", mfc="none", mec="#333", label="Property-type median"),
        Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="Recommended: linear / ridge"),
        Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="Recommended: random forest"),
        Line2D([], [], marker="o", ls="", color=GRAY, ms=8, mec="#000", label="Black outline: CI says better than general model")],
        frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(.45, -.08), ncol=3)
    plt.tight_layout()
    plt.savefig(FIG / "report_fig8_portfolio.png")
    plt.close()


def fig9(cand):
    lo = .5
    y = cand["site_eui"].to_numpy()
    gen = np.clip(cand["gen_ref"].to_numpy(), lo, None)
    port = np.clip(cand["portfolio_pred"].to_numpy(), lo, None)
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.8))

    def hexpanel(a, pred, title, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        hb = a.hexbin(np.log10(pred[m]), np.log10(np.clip(y[m], lo, None)), gridsize=45, bins="log", mincnt=1, cmap="Blues",
                      extent=(-.3, 3.3, -.3, 3.3))
        a.plot([-.3, 3.3], [-.3, 3.3], color=ORANGE, lw=1.1)
        ticks = [1, 10, 100, 1000]
        a.set_xticks(np.log10(ticks)); a.set_xticklabels(ticks)
        a.set_yticks(np.log10(ticks)); a.set_yticklabels(ticks)
        a.set_xlabel("Predicted Site EUI (kBtu/ft², log)")
        a.set_ylabel("Actual Site EUI (kBtu/ft², log)")
        a.set_title(title, loc="left", fontsize=9)
        a.set_xlim(-.3, 3.3); a.set_ylim(-.3, 3.3)
        r2v = r2(y[m], pred[m])
        a.text(.03, .95, f"R² = {r2v:.2f}\nMAE = {np.abs(y[m] - pred[m]).mean():.1f}", transform=a.transAxes, va="top", fontsize=8)

    hexpanel(ax[0, 0], gen, "General model, all buildings")
    hexpanel(ax[0, 1], port, "Recommended portfolio, all buildings")
    res_g, res_p = y - cand["gen_ref"].to_numpy(), y - cand["portfolio_pred"].to_numpy()
    bins = np.linspace(-120, 120, 80)
    ax[0, 2].hist(res_g, bins=bins, color=GRAY, alpha=.75, label=f"General (median {np.median(res_g):+.1f})")
    ax[0, 2].hist(res_p, bins=bins, color=ORANGE, alpha=.6, label=f"Portfolio (median {np.median(res_p):+.1f})")
    ax[0, 2].axvline(0, color="#333", lw=.8)
    ax[0, 2].set_xlabel("Residual = actual − predicted (kBtu/ft²)")
    ax[0, 2].set_ylabel("Building-years")
    ax[0, 2].set_title("Residuals (clipped at ±120)", loc="left", fontsize=9)
    ax[0, 2].legend(frameon=False, fontsize=8)
    for a, t in zip(ax[1], ["Multifamily Housing", "Office", "Non-Refrigerated Warehouse"]):
        hexpanel(a, port, f"{short(t)}, portfolio model", (cand["property_type"] == t).to_numpy())
    plt.tight_layout()
    plt.savefig(FIG / "report_fig9_pred_vs_actual.png")
    plt.close()


def fig10():
    gen = pd.read_csv(OUT / "feature_importance_general.csv")
    bt = pd.read_csv(OUT / "feature_importance_by_type.csv")
    single = gen[~gen["is_group"]].sort_values("mae_increase")
    wg = gen[gen["is_group"]].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    yy = np.arange(len(single))
    ax[0].barh(yy, single["mae_increase"], xerr=single["mae_increase_sd_across_folds"].fillna(0), color=BLUE, height=.7,
               error_kw={"ecolor": "#555", "lw": .8, "capsize": 2})
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels(single["label"])
    ax[0].axvline(0, color="#999", lw=.6)
    ax[0].set_xlabel("Increase in MAE when the feature is shuffled (kBtu/ft²)")
    ax[0].set_title("General model: what the prediction relies on", loc="left")
    ax[0].text(.98, .03, f"All 5 weather variables shuffled together:\n+{wg['mae_increase']:.2f} kBtu/ft² "
                         f"({wg['pct_of_baseline_mae']:.1f}% of MAE)", transform=ax[0].transAxes, ha="right", fontsize=8,
               bbox=dict(boxstyle="round", fc="#f3f4f6", ec="none"))
    feats = [f for f in ["floor_area", "building_age", "latitude", "longitude", "year", "weather_group"]]
    types = bt["scope"].unique().tolist()
    mat = np.array([[bt[(bt["scope"] == t) & (bt["feature"] == f)]["pct_of_baseline_mae"].mean() for f in feats] for t in types])
    im = ax[1].imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    ax[1].set_xticks(range(len(feats)))
    ax[1].set_xticklabels([gen.set_index("feature")["label"].get(f, f) for f in feats], rotation=30, ha="right")
    ax[1].set_yticks(range(len(types)))
    ax[1].set_yticklabels([short(t) for t in types])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax[1].text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8,
                       color="white" if mat[i, j] > .6 * np.nanmax(mat) else "#222")
    ax[1].set_title("Within each type: MAE increase, % of that type's MAE", loc="left")
    for s in ax[1].spines.values():
        s.set_visible(False)
    plt.colorbar(im, ax=ax[1], fraction=.04, pad=.02).outline.set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "report_fig10_feature_importance.png")
    plt.close()


def selection(cand, pred_col, n_years=3):
    """Audit selection as in 07: three usable years, >=2 positive residuals, rank by mean excess kBtu."""
    d = cand.assign(res=cand["site_eui"] - cand[pred_col])
    d["excess"] = d["res"] * d["floor_area"]
    b = d.groupby("building_id").agg(years=("year", "nunique"), pos=("res", lambda x: int((x > 0).sum())),
                                     excess=("excess", "mean"), ptype=("property_type", "first"),
                                     actual=("site_eui", "mean"), pred=(pred_col, "mean"))
    three = b[b["years"] == n_years].copy()
    el = three[three["pos"] >= 2].sort_values("excess", ascending=False)
    top = set(el.head(int(np.ceil(len(el) * .10))).index)
    three["eligible"] = three.index.isin(el.index)
    three["top_decile"] = three.index.isin(top)
    three["ratio_gt5"] = (three["actual"] / three["pred"]) > 5
    return three


def fairness(cand, portfolio):
    big = portfolio.loc[portfolio["segment"] != "All other types", "segment"].tolist()
    seg = np.where(cand["property_type"].isin(big), cand["property_type"], "All other types")
    cand = cand.assign(segment=seg)
    rows = []
    sel = {k: selection(cand, c) for k, c in [("general", "gen_ref"), ("portfolio", "portfolio_pred")]}
    seg_of = cand.groupby("building_id")["segment"].first()
    for k in sel:
        sel[k]["segment"] = seg_of.reindex(sel[k].index).to_numpy()
    for s, sub in cand.groupby("segment"):
        r = {"segment": s, "n_rows": len(sub), "n_buildings": sub["building_id"].nunique(), "mean_site_eui": sub["site_eui"].mean()}
        for k, c in [("general", "gen_ref"), ("portfolio", "portfolio_pred")]:
            res = sub["site_eui"] - sub[c]
            r.update({f"{k}_mae": res.abs().mean(), f"{k}_mae_pct_of_mean_eui": 100 * res.abs().mean() / sub["site_eui"].mean(),
                      f"{k}_bias": res.mean(), f"{k}_median_residual": res.median(), f"{k}_share_positive": (res > 0).mean()})
            t = sel[k][sel[k]["segment"] == s]
            r.update({f"{k}_three_year_buildings": len(t), f"{k}_eligible_rate": t["eligible"].mean() if len(t) else np.nan,
                      f"{k}_top_decile_rate": t["top_decile"].mean() if len(t) else np.nan})
        rows.append(r)
    d = pd.DataFrame(rows)
    for k in sel:
        overall = sel[k]["top_decile"].mean()
        overall_el = sel[k]["eligible"].mean()
        d[f"{k}_top_decile_disparity"] = d[f"{k}_top_decile_rate"] / overall
        d[f"{k}_eligible_disparity"] = d[f"{k}_eligible_rate"] / overall_el
        d.attrs[f"{k}_overall_top_decile_rate"] = overall
        d.attrs[f"{k}_overall_eligible_rate"] = overall_el
    d["small_sample"] = d["n_buildings"] < 100
    d = d.sort_values("n_rows", ascending=False).reset_index(drop=True)
    d.to_csv(OUT / "fairness_by_type.csv", index=False)
    return d


def fig11(d):
    d = d[d["segment"] != "All other types"].head(12).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    h = .38
    for a, (gc, pc, title, xl) in zip(ax, [
        ("general_mae_pct_of_mean_eui", "portfolio_mae_pct_of_mean_eui", "Relative error", "MAE as % of the type's mean EUI"),
        ("general_bias", "portfolio_bias", "Bias (mean actual − predicted)", "kBtu/ft²; + means under-predicted"),
        ("general_top_decile_disparity", "portfolio_top_decile_disparity", "Audit top-decile selection", "Rate relative to all buildings (1 = parity)")]):
        a.barh(y + h / 2, d[gc], h, color=GRAY, label="General model")
        a.barh(y - h / 2, d[pc], h, color=ORANGE, label="Recommended portfolio")
        a.set_xlabel(xl)
        a.set_title(title, loc="left")
        a.axvline(0 if "bias" in gc else (1 if "disparity" in gc else 0), color="#333", lw=.8, ls="--" if "disparity" in gc else "-")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{short(s)} (n={int(n):,})" for s, n in zip(d["segment"], d["n_buildings"])], fontsize=8)
    ax[2].legend(frameon=False, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG / "report_fig11_fairness.png")
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cand, per_type, portfolio, grouping = load()
    fig6(cand, per_type, grouping)
    fig7(grouping)
    fig8(portfolio)
    fig9(cand)
    if (OUT / "feature_importance_general.csv").exists():
        fig10()
    d = fairness(cand, portfolio)
    fig11(d)
    print(f"Wrote figures to {FIG} and {OUT / 'fairness_by_type.csv'}")


if __name__ == "__main__":
    main()
