"""Regenerate every figure and report statistic from the pipeline outputs.

Run after scripts 01-08. Reads only files already in data/ and outputs/, trains
nothing, and writes:
  outputs/figures/report_fig1_eui_distribution.png
  outputs/figures/report_fig2_correlations.png
  outputs/figures/report_fig3_model_mae.png
  outputs/figures/report_fig4_residual_persistence.png
  outputs/figures/report_fig5_ranking.png
  outputs/report_stats.json   (every number quoted in the report)

All labels are computed from the data, so figures stay correct when the
portal data changes and the pipeline is rerun.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config

OUT = ROOT / "outputs"
FIG = OUT / "figures"
BLUE, ORANGE, GRAY, RED = "#2b6cb0", "#dd7a1f", "#8a8f98", "#c0392b"
N_BOOT = 1000

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 200,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#333", "ytick.color": "#333",
})


def read_json(name):
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def load():
    o = pd.read_csv(OUT / "oof_predictions.csv", dtype={"building_id": str})
    o["res"] = o["site_eui"] - o["predicted_site_eui"]
    r = pd.read_csv(OUT / "audit_priority_ranking.csv", dtype={"building_id": str})
    if "actual_to_predicted_ratio" not in r:  # rankings made before the review flag existed
        r["actual_to_predicted_ratio"] = r["mean_actual_site_eui"] / r["mean_predicted_site_eui"]
        r["check_data_first"] = r["actual_to_predicted_ratio"] > 5
    metrics = pd.read_csv(OUT / "model_metrics.csv")
    return o, r, metrics


def paired_bootstrap_mae_diff(o, with_col, without_col, seed):
    """Building-level bootstrap of MAE(with) - MAE(without); negative means weather helped."""
    d = (o["site_eui"] - o[with_col]).abs() - (o["site_eui"] - o[without_col]).abs()
    g = d.groupby(o["building_id"]).agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    est = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"mae_diff": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def compute_stats(o, r, metrics, cfg):
    stats = {}
    stats["rows"] = {"modeled_rows": len(o), "modeled_buildings": int(o["building_id"].nunique()),
                     "rows_by_year": {int(k): int(v) for k, v in o.groupby("year").size().items()}}
    per_b = o.groupby("building_id")["year"].nunique()
    stats["buildings_by_years_present"] = {int(k): int(v) for k, v in per_b.value_counts().sort_index().items()}
    for col in ["site_eui", "floor_area", "building_age", "emissions", "station_distance_miles"]:
        s = o[col]
        stats.setdefault("variable_summary", {})[col] = {
            "mean": float(s.mean()), "std": float(s.std()), "median": float(s.median()),
            "p95": float(s.quantile(.95)), "max": float(s.max()), "min": float(s.min())}
    stats["site_eui_skewness"] = float(o["site_eui"].skew())
    stats["median_site_eui_by_year"] = {int(k): float(v) for k, v in o.groupby("year")["site_eui"].median().items()}
    stats["property_types"] = int(o["property_type"].nunique())
    top = o["property_type"].value_counts().head(10)
    stats["top_property_types"] = [
        {"type": t, "rows": int(n), "share": float(n / len(o)),
         "median_site_eui": float(o.loc[o["property_type"] == t, "site_eui"].median())}
        for t, n in top.items()]
    stats["rows_by_station"] = {k: int(v) for k, v in o["station_id"].value_counts().items()}

    # Weather benefit (does adding weather change grouped out-of-fold error?)
    seed = cfg["random_state"]
    stats["weather_effect_bootstrap"] = {
        "random_forest": paired_bootstrap_mae_diff(o, "pred_rf_weather", "pred_rf_building", seed),
        "linear": paired_bootstrap_mae_diff(o, "pred_linear_weather", "pred_linear_building", seed),
        "n_bootstrap": N_BOOT}

    # Residual behaviour
    stats["residuals"] = {"mean": float(o["res"].mean()), "median": float(o["res"].median()),
                          "share_positive": float((o["res"] > 0).mean())}
    typ = o["property_type"].value_counts().head(8).index
    stats["median_residual_by_type"] = {t: float(o.loc[o["property_type"] == t, "res"].median()) for t in typ}
    w = o.pivot_table(index="building_id", columns="year", values="res").dropna()
    stats["residual_persistence"] = {
        "three_year_buildings": len(w),
        "pearson": w.corr().round(4).to_dict(), "spearman": w.corr(method="spearman").round(4).to_dict(),
        "share_same_sign_all_years": float((((w > 0).sum(axis=1)) % len(w.columns) == 0).mean())}
    pos = o[o["building_id"].isin(w.index)].groupby("building_id")["res"].apply(lambda x: int((x > 0).sum()))
    stats["positive_years_distribution"] = {int(k): int(v) for k, v in pos.value_counts().sort_index().items()}

    # Ranking
    n10 = int(np.ceil(len(r) * .10))
    total = r["mean_annual_excess_kbtu"].sum()
    topd = r.head(n10)
    stats["ranking"] = {
        "eligible": len(r), "eligible_share_of_three_year": len(r) / len(w),
        "top_decile_size": n10,
        "top100_excess_share": float(r.head(100)["mean_annual_excess_kbtu"].sum() / total),
        "top_decile_excess_share": float(topd["mean_annual_excess_kbtu"].sum() / total),
        "eligible_negative_mean_excess": int((r["mean_annual_excess_kbtu"] < 0).sum()),
        "flagged_total": int(r["check_data_first"].sum()),
        "flagged_in_top_decile": int(topd["check_data_first"].sum()),
        "median_floor_area_top_decile": float(topd["floor_area"].median()),
        "median_floor_area_eligible": float(r["floor_area"].median()),
        "multifamily_share_eligible": float((r["property_type"] == "Multifamily Housing").mean()),
        "multifamily_share_modeled_rows": float((o["property_type"] == "Multifamily Housing").mean()),
        "top10": r.head(10)[["priority_rank", "property_type", "floor_area", "mean_actual_site_eui",
                             "mean_predicted_site_eui", "mean_annual_excess_kbtu",
                             "actual_to_predicted_ratio", "check_data_first"]].to_dict("records")}

    # Other reports, passed through for one-stop reference
    stats["model_metrics"] = metrics.to_dict("records")
    stats["ranking_stability"] = pd.read_csv(OUT / "model_ranking_stability.csv").to_dict("records")
    stats["cluster_selection"] = pd.read_csv(OUT / "cluster_selection.csv").to_dict("records")
    stats["cluster_summary"] = pd.read_csv(OUT / "cluster_summary.csv").to_dict("records")
    for name in ["data_quality_report", "join_report", "geocoding_report"]:
        stats[name] = read_json(f"{name}.json")
    corr_cols = ["site_eui", "floor_area", "building_age", "latitude", "longitude", "mean_temp_f",
                 "total_prcp_mm", "hdd65", "cdd65", "hot_days_90f", "emissions"]
    stats["spearman_with_site_eui"] = {c: float(v) for c, v in
                                       o[corr_cols].corr(method="spearman")["site_eui"].items()}
    return stats


def fig1(o):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
    bins = np.logspace(np.log10(max(o["site_eui"].min(), .1)), np.log10(o["site_eui"].max()), 60)
    ax[0].hist(o["site_eui"], bins=bins, color=BLUE, alpha=.85)
    ax[0].set_xscale("log")
    med = o["site_eui"].median()
    ax[0].axvline(med, color=ORANGE, lw=1.5)
    ax[0].text(med * 1.15, ax[0].get_ylim()[1] * .92, f"median {med:.1f}", color=ORANGE, fontsize=8)
    ax[0].set_xlabel("Site EUI (kBtu/ft², log scale)"); ax[0].set_ylabel("Building-years")
    ax[0].set_title("Site EUI is heavily right-skewed")
    top = o["property_type"].value_counts().head(10).index
    m = o[o["property_type"].isin(top)].groupby("property_type")["site_eui"].median().sort_values()
    cnt = o["property_type"].value_counts()
    ax[1].barh(m.index, m.values, color=BLUE, alpha=.85)
    for i, (k, v) in enumerate(m.items()):
        ax[1].text(v + .01 * m.max(), i, f"{v:.1f}  (n={cnt[k]:,})", va="center", fontsize=7.5, color="#333")
    ax[1].set_xlim(0, m.max() * 1.4)
    ax[1].set_xlabel("Median Site EUI (kBtu/ft²)"); ax[1].set_title("Median EUI, ten most common types")
    ax[1].tick_params(axis="y", labelsize=8)
    plt.tight_layout(); plt.savefig(FIG / "report_fig1_eui_distribution.png"); plt.close()


def fig2(o):
    cols = ["site_eui", "floor_area", "building_age", "latitude", "longitude", "mean_temp_f",
            "total_prcp_mm", "hdd65", "cdd65", "hot_days_90f", "emissions"]
    lab = ["Site EUI", "Floor area", "Building age", "Latitude", "Longitude", "Mean temp",
           "Precip", "HDD65", "CDD65", "Days >90°F", "GHG emissions"]
    c = o[cols].corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(c, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    for i in range(len(lab)):
        for j in range(len(lab)):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(c[i, j]) > .6 else "#222")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Spearman correlations ({len(o):,} building-years)")
    plt.colorbar(im, fraction=.04, pad=.02).outline.set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "report_fig2_correlations.png"); plt.close()


def fig3(metrics):
    order = ["overall_median", "property_type_median", "linear_building", "linear_weather", "rf_building", "rf_weather"]
    names = {"overall_median": "Overall median", "property_type_median": "Property-type median",
             "linear_building": "Linear, building only", "linear_weather": "Linear + weather",
             "rf_building": "Random forest, building only", "rf_weather": "Random forest + weather"}
    oof = metrics[metrics["scope"] == "grouped_oof_all_buildings"].set_index("model").loc[order]
    ho = metrics[metrics["scope"] == "fixed_20pct_heldout_buildings"].set_index("model")
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(order))[::-1].astype(float)
    h = .34
    best = oof["MAE"].idxmin()
    for yi, m in zip(y, order):
        c = GRAY if m == "overall_median" else ORANGE if m == best else BLUE
        ax.barh(yi + h / 2, oof.loc[m, "MAE"], height=h, color=c)
        ax.text(oof.loc[m, "MAE"] + .3, yi + h / 2, f"{oof.loc[m, 'MAE']:.1f}", va="center", fontsize=8.5, fontweight="bold")
        if m in ho.index:
            ax.barh(yi - h / 2, ho.loc[m, "MAE"], height=h, color=c, alpha=.45)
            ax.text(ho.loc[m, "MAE"] + .3, yi - h / 2, f"{ho.loc[m, 'MAE']:.1f}", va="center", fontsize=8, color="#444")
    ax.set_yticks(y); ax.set_yticklabels([names[m] for m in order]); ax.set_xlim(0, oof["MAE"].max() * 1.15)
    ax.set_xlabel("Mean absolute error (kBtu/ft²)")
    ax.set_title(f"Lowest MAE: {names[best]}", loc="left")
    ax.legend(handles=[Patch(color=BLUE, label="Grouped 5-fold out-of-fold, all buildings"),
                       Patch(color=BLUE, alpha=.45, label="Fixed 20% held-out buildings")],
              loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2, frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "report_fig3_model_mae.png", bbox_inches="tight"); plt.close()


def fig4(o, r):
    w = o.pivot_table(index="building_id", columns="year", values="res").dropna()
    years = sorted(w.columns)
    a, b = years[0], years[1]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
    ax[0].scatter(w[a], w[b], s=5, alpha=.25, color=BLUE, edgecolors="none")
    ax[0].set_xscale("symlog", linthresh=10); ax[0].set_yscale("symlog", linthresh=10)
    lim = [min(w[a].min(), w[b].min()) * 1.1, max(w[a].max(), w[b].max()) * 1.1]
    lim[0] = max(lim[0], -300)
    ax[0].plot(lim, lim, color=ORANGE, lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].axhline(0, color="#999", lw=.6); ax[0].axvline(0, color="#999", lw=.6)
    ax[0].set_xlabel(f"{a} residual (kBtu/ft², symlog)"); ax[0].set_ylabel(f"{b} residual (kBtu/ft², symlog)")
    ax[0].set_title(f"Residuals repeat across years (r = {w.corr().loc[a, b]:.2f})")
    npos = o[o["building_id"].isin(w.index)].groupby("building_id")["res"].apply(lambda x: int((x > 0).sum()))
    npos = npos.value_counts().reindex(range(len(years) + 1), fill_value=0).sort_index()
    need = 2
    ax[1].bar(npos.index.astype(str), npos.values, color=[BLUE if k >= need else GRAY for k in npos.index], alpha=.9)
    for i, v in zip(npos.index, npos.values):
        ax[1].text(i, v + npos.max() * .015, f"{v:,}", ha="center", fontsize=8.5)
    ax[1].set_xlabel(f"Years with a positive residual (of {len(years)})"); ax[1].set_ylabel("Buildings")
    ax[1].set_ylim(0, npos.max() * 1.18)
    ax[1].set_title(f"Eligible for ranking: {need}+ positive years")
    ax[1].text(len(years) - .5, npos.max() * 1.02, f"{len(r):,} eligible", ha="center", color=BLUE, fontsize=9, fontweight="bold")
    plt.tight_layout(); plt.savefig(FIG / "report_fig4_residual_persistence.png"); plt.close()


def fig5(o, r):
    n10 = int(np.ceil(len(r) * .10))
    topd = r.head(n10)
    groups = ["Multifamily Housing", "Office", "Manufacturing/Industrial Plant",
              "Non-Refrigerated Warehouse", "Distribution Center", "Retail Store"]

    def share(s):
        v = s.value_counts(normalize=True)
        d = {g: v.get(g, 0) for g in groups}
        d["All other types"] = 1 - sum(d.values())
        return d

    df = pd.DataFrame({"All modeled\nbuilding-years": share(o["property_type"]),
                       f"Eligible for\nranking ({len(r):,})": share(r["property_type"]),
                       f"Top decile\n({n10})": share(topd["property_type"])}).T
    pal = ["#2b6cb0", "#4f9bd9", "#dd7a1f", "#e9b872", "#7a8c3c", "#b05fa5", "#c8ccd2"]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    left = np.zeros(len(df))
    for g, col in zip(df.columns, pal):
        ax[0].barh(df.index, df[g].values * 100, left=left * 100, color=col, label=g, height=.6)
        for i, (v, l) in enumerate(zip(df[g].values, left)):
            if v > .07:
                ax[0].text((l + v / 2) * 100, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        left += df[g].values
    ax[0].invert_yaxis(); ax[0].set_xlim(0, 100); ax[0].set_xlabel("Share (%)"); ax[0].set_title("Who is in the ranking")
    ax[0].legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=7)
    flag = r["check_data_first"]
    ratio = r["actual_to_predicted_ratio"]
    bins = np.logspace(np.log10(min(.3, ratio.min())), np.log10(max(70, ratio.max())), 55)
    ax[1].hist(ratio[~flag], bins=bins, color=BLUE, alpha=.9, label=f"Not flagged ({int((~flag).sum()):,})")
    ax[1].hist(ratio[flag], bins=bins, color=RED, alpha=.95, label=f"Flagged, ratio > 5 ({int(flag.sum())})")
    ax[1].set_xscale("log"); ax[1].axvline(5, color="#333", ls="--", lw=1)
    ax[1].set_xticks([.5, 1, 2, 5, 10, 20, 50]); ax[1].set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax[1].set_xlabel("Actual / predicted mean Site EUI (log scale)"); ax[1].set_ylabel("Eligible buildings")
    ax[1].set_title(f"Most candidates sit near {ratio.median():.2f}×"); ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "report_fig5_ranking.png", bbox_inches="tight"); plt.close()


def main():
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    o, r, metrics = load()
    fig1(o); fig2(o); fig3(metrics); fig4(o, r); fig5(o, r)
    stats = compute_stats(o, r, metrics, cfg)
    with open(OUT / "report_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"Wrote 5 figures to {FIG} and statistics to {OUT / 'report_stats.json'}")


if __name__ == "__main__":
    main()
