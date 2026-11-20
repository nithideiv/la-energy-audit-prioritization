"""Building-level audit evidence cards: why does this building qualify for an audit?

For one building, show what a manager would see: reported Site EUI each year next to the
EUI expected for a building like it (the recommended model for its property type, scored
out of fold, so the building's own data never trained the prediction), the typical error
band for that property type, and the same-year peer median when the peer comparison has one.

A building qualifies for the screen when it has usable records in all three years, reported
EUI above expected in at least two of them, and a positive mean excess. That is the rule in
07_model_and_rank.py plus a positive mean excess. "Well above" means more than one typical
error (the property type's out-of-fold MAE) above expected. A qualifying building is a
candidate for a data check and then an audit, not a finding of waste.

Usage:
  python scripts/14_building_audit_cards.py                    # a few example cards + ranking file
  python scripts/14_building_audit_cards.py --building-id 10353

Writes:
  outputs/audit_priority_ranking_portfolio.csv   the same screen using the portfolio predictions
  outputs/portfolio_ranking_comparison.json      overlap with the ranking from 07
  outputs/figures/building_cards/<building_id>_audit_card.png
"""
from pathlib import Path
import argparse
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CARDS = OUT / "figures" / "building_cards"
BLUE, LIGHT_BLUE, ORANGE, GRAY, RED, GREEN = "#2b6cb0", "#a9c3e4", "#dd7a1f", "#8a8f98", "#c0392b", "#4a8a5b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 200})


def load():
    cand = pd.read_csv(OUT / "per_type_oof_predictions.csv", dtype={"building_id": str})
    base = pd.read_csv(OUT / "oof_predictions.csv", dtype={"building_id": str})
    assert (cand["building_id"].to_numpy() == base["building_id"].to_numpy()).all()
    for c in ["floor_area", "postal_code"]:
        cand[c] = base[c].to_numpy()
    portfolio = pd.read_csv(OUT / "model_portfolio.csv").set_index("segment")
    big = set(portfolio.index) - {"All other types"}
    cand["segment"] = np.where(cand["property_type"].isin(big), cand["property_type"], "All other types")
    cand["expected"] = cand["portfolio_pred"].clip(lower=0)
    cand["residual"] = cand["site_eui"] - cand["expected"]
    cand["typical_error"] = cand["segment"].map(portfolio["mae_recommended"])
    cand["above"] = cand["residual"] > 0
    cand["well_above"] = cand["residual"] > cand["typical_error"]
    cand["excess_kbtu"] = cand["residual"] * cand["floor_area"]
    peers_path = OUT / "peer_comparisons_annual.csv"
    if peers_path.exists():
        peers = pd.read_csv(peers_path, dtype={"building_id": str})[["building_id", "year", "peer_median_site_eui", "peer_count"]]
        cand = cand.merge(peers, on=["building_id", "year"], how="left")
    else:
        cand["peer_median_site_eui"] = np.nan
        cand["peer_count"] = np.nan
    return cand, portfolio


def building_table(cand, n_years=3):
    b = cand.groupby("building_id").agg(
        years=("year", "nunique"), positive_years=("above", "sum"), well_above_years=("well_above", "sum"),
        mean_annual_excess_kbtu=("excess_kbtu", "mean"), mean_actual_site_eui=("site_eui", "mean"),
        mean_expected_site_eui=("expected", "mean"), mean_residual=("residual", "mean"),
        property_type=("property_type", "first"), segment=("segment", "first"),
        floor_area=("floor_area", "median"), postal_code=("postal_code", "first")).reset_index()
    b["actual_to_expected_ratio"] = b["mean_actual_site_eui"] / b["mean_expected_site_eui"].clip(lower=.5)
    b["check_data_first"] = b["actual_to_expected_ratio"] > 5
    b["qualifies"] = (b["years"] == n_years) & (b["positive_years"] >= 2) & (b["mean_annual_excess_kbtu"] > 0)
    three = b[b["years"] == n_years]
    b["percentile_in_segment"] = np.nan
    pct = three.groupby("segment")["mean_residual"].rank(pct=True) * 100
    b.loc[three.index, "percentile_in_segment"] = pct
    ranked = b[b["qualifies"]].sort_values("mean_annual_excess_kbtu", ascending=False).copy()
    ranked.insert(0, "priority_rank", range(1, len(ranked) + 1))
    b = b.merge(ranked[["building_id", "priority_rank"]], on="building_id", how="left")
    return b, ranked


def compare_with_original(ranked):
    path = OUT / "audit_priority_ranking.csv"
    if not path.exists():
        return {}
    old = pd.read_csv(path, dtype={"building_id": str})
    n_old, n_new = int(np.ceil(len(old) * .1)), int(np.ceil(len(ranked) * .1))
    a, c = set(old.head(n_old)["building_id"]), set(ranked.head(n_new)["building_id"])
    return {"original_eligible": len(old), "portfolio_eligible": len(ranked),
            "eligible_overlap_jaccard": len(set(old["building_id"]) & set(ranked["building_id"])) / len(set(old["building_id"]) | set(ranked["building_id"])),
            "top_decile_original": n_old, "top_decile_portfolio": n_new,
            "top_decile_jaccard": len(a & c) / len(a | c),
            "flagged_check_data_first_portfolio": int(ranked["check_data_first"].sum())}


def card(cand, table, portfolio, building_id):
    rows = cand[cand["building_id"] == building_id].sort_values("year")
    if rows.empty:
        raise SystemExit(f"building_id {building_id} not found among modeled buildings")
    b = table[table["building_id"] == building_id].iloc[0]
    seg = b["segment"]
    model_label = portfolio.loc[seg, "recommended_label"]
    n_seg = int(table[(table["segment"] == seg) & (table["years"] == 3)].shape[0])
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    years = rows["year"].astype(int).tolist()
    x = np.arange(len(years))
    w = .36
    a.bar(x - w / 2, rows["expected"], w, color=LIGHT_BLUE, yerr=rows["typical_error"], error_kw={"ecolor": "#555", "lw": 1, "capsize": 3},
          label="Expected for a building like this")
    colors = [RED if wa else ORANGE if ab else GREEN for wa, ab in zip(rows["well_above"], rows["above"])]
    a.bar(x + w / 2, rows["site_eui"], w, color=colors)
    top = max(rows["site_eui"].max(), (rows["expected"] + rows["typical_error"]).max())
    for xi, (e, s) in enumerate(zip(rows["expected"], rows["site_eui"])):
        a.text(xi - w / 2, 0 + top * .012, f"{e:.0f}", ha="center", va="bottom", fontsize=8, color="#123")
        a.text(xi + w / 2, s + top * .015, f"{s:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    if rows["peer_median_site_eui"].notna().any():
        a.scatter(x + w / 2 + .0, rows["peer_median_site_eui"], marker="D", s=34, color="#111", zorder=5, label="Median of 10 similar nearby buildings")
    a.set_xticks(x)
    a.set_xticklabels(years)
    a.set_ylabel("Site EUI (kBtu/ft²)")
    a.set_ylim(0, top * 1.12)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color=LIGHT_BLUE, label="Expected for a building like this (± typical error)"),
               Patch(color=RED, label="Reported: well above expected"), Patch(color=ORANGE, label="Reported: above expected"),
               Patch(color=GREEN, label="Reported: at or below expected")]
    if rows["peer_median_site_eui"].notna().any():
        handles.append(Line2D([], [], marker="D", ls="", color="#111", label="Similar-building median"))
    a.legend(handles=handles, frameon=False, fontsize=7.3, loc="upper center", bbox_to_anchor=(.5, -.09), ncol=2)
    a.set_title(f"Building {building_id}: reported vs. expected energy use", loc="left", fontsize=10)

    t = ax[1]
    t.axis("off")
    zip_txt = f", ZIP {int(b['postal_code'])}" if pd.notna(b["postal_code"]) else ""
    t.text(0, 1.0, f"{b['property_type']}, {b['floor_area']:,.0f} ft²{zip_txt}", fontsize=9.5, fontweight="bold", va="top")
    if b["qualifies"]:
        head, col = "Meets the audit screen", RED
        lines = [
            f"Reported EUI was above expected in {int(b['positive_years'])} of {int(b['years'])} years, "
            f"and well above (more than the typical error) in {int(b['well_above_years'])}.",
            f"Average: {b['mean_actual_site_eui']:.0f} reported vs. {b['mean_expected_site_eui']:.0f} expected kBtu/ft² "
            f"({100 * (b['actual_to_expected_ratio'] - 1):+.0f}%).",
            f"That gap is about {b['mean_annual_excess_kbtu'] / 1000:,.0f} MMBtu per year.",
            f"Rank {int(b['priority_rank']):,} of {int(table['qualifies'].sum()):,} buildings that meet the screen; "
            f"higher gap than {b['percentile_in_segment']:.0f}% of {n_seg:,} three-year buildings in this property-type group.",
        ]
        if b["check_data_first"]:
            lines.append("Reported use is more than 5x expected. Check the utility data and the reporting boundary before anything else.")
    else:
        head, col = "Does not meet the audit screen", GREEN
        why = []
        if b["years"] < 3:
            why.append(f"only {int(b['years'])} of 3 years have usable data")
        if b["positive_years"] < 2:
            why.append(f"reported EUI was above expected in only {int(b['positive_years'])} of {int(b['years'])} years")
        if b["mean_annual_excess_kbtu"] <= 0:
            why.append("average use is at or below expected")
        lines = [f"Reason: {'; '.join(why)}.",
                 f"Average: {b['mean_actual_site_eui']:.0f} reported vs. {b['mean_expected_site_eui']:.0f} expected kBtu/ft²."]
    t.text(0, .91, head, fontsize=12, fontweight="bold", color=col, va="top")
    ypos = .82
    for ln in lines:
        wrapped = textwrap.fill(ln, 62)
        t.text(0, ypos, wrapped, fontsize=8.6, va="top")
        ypos -= .062 * (wrapped.count("\n") + 1) + .03
    note = (f"Expected use comes from the {model_label.lower()} model for this property type, trained on other "
            "buildings only. A gap is a reason to look, not proof of waste: occupancy hours, process loads or reporting "
            "errors can explain it.")
    t.text(0, .0, textwrap.fill(note, 74), fontsize=7.4, color="#555", va="bottom")
    plt.tight_layout()
    CARDS.mkdir(parents=True, exist_ok=True)
    path = CARDS / f"{building_id}_audit_card.png"
    plt.savefig(path)
    plt.close()
    return path


def pick_examples(table, ranked):
    ok = ranked[~ranked["check_data_first"] & (ranked["floor_area"] >= 20000)]
    picks = []
    for t in ["Multifamily Housing", "Office", "Non-Refrigerated Warehouse"]:
        s = ok[ok["property_type"] == t]
        if len(s):
            picks.append(s.iloc[min(2, len(s) - 1)]["building_id"])      # near the top, but not the single most extreme one
    neg = table[(table["years"] == 3) & ~table["qualifies"] & (table["property_type"] == "Multifamily Housing")
                & (table["mean_residual"] < 0)]
    if len(neg):
        picks.append(neg.iloc[len(neg) // 2]["building_id"])
    flagged = ranked[ranked["check_data_first"]]
    if len(flagged):
        picks.append(flagged.iloc[0]["building_id"])
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building-id", action="append", help="building to draw a card for (repeatable)")
    args = ap.parse_args()
    cand, portfolio = load()
    table, ranked = building_table(cand)
    ranked.to_csv(OUT / "audit_priority_ranking_portfolio.csv", index=False)
    cmp = compare_with_original(ranked)
    (OUT / "portfolio_ranking_comparison.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print("Ranking with portfolio predictions:", json.dumps(cmp))
    ids = args.building_id or pick_examples(table, ranked)
    for bid in ids:
        print("wrote", card(cand, table, portfolio, str(bid)))


if __name__ == "__main__":
    main()
