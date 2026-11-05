"""Compare each eligible building with nearby peers that share its use and year.

Peers are selected without using energy use: the matching features are log floor
area, age, location, and occupancy. The result is a screening list, not proof
that a building wastes energy.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config

FEATURES = ["log_floor_area", "building_age", "latitude", "longitude", "occupancy"]


def peer_eligibility(df: pd.DataFrame, cap: float) -> pd.Series:
    """Rows that have a comparable, stable one-building reporting boundary."""
    required = df[["site_eui", "floor_area", "building_age", "latitude", "longitude", "occupancy", "property_type"]].notna().all(axis=1)
    return (
        required
        & df["site_eui"].gt(0)
        & df["site_eui"].le(cap)
        & df["floor_area"].ge(1000)
        & ~df["needs_data_review"].fillna(True)
    )


def compare_group(group: pd.DataFrame, k: int, min_peers: int) -> pd.DataFrame:
    """Return leave-one-out peer summaries for one property-type/year group."""
    group = group.copy()
    if len(group) - 1 < min_peers:
        return pd.DataFrame()
    x = StandardScaler().fit_transform(group[FEATURES].to_numpy())
    neighbor_count = min(k + 1, len(group))
    distances, positions = NearestNeighbors(n_neighbors=neighbor_count).fit(x).kneighbors(x)
    rows = []
    for row_pos, (d, p) in enumerate(zip(distances, positions)):
        chosen = [(distance, index) for distance, index in zip(d, p) if index != row_pos][:k]
        if len(chosen) < min_peers:
            continue
        peer_pos = [index for _, index in chosen]
        peer_eui = group.iloc[peer_pos]["site_eui"]
        peer_median = peer_eui.median()
        row = group.iloc[row_pos]
        rows.append({
            "building_id": row["building_id"], "year": row["year"],
            "property_type": row["property_type"], "site_eui": row["site_eui"],
            "floor_area": row["floor_area"], "peer_count": len(peer_pos),
            "peer_median_site_eui": peer_median,
            "peer_mean_site_eui": peer_eui.mean(),
            "peer_eui_excess": row["site_eui"] - peer_median,
            "actual_to_peer_median_ratio": row["site_eui"] / peer_median if peer_median > 0 else np.nan,
            "peer_distance_mean": float(np.mean([distance for distance, _ in chosen])),
            "peer_distance_max": float(np.max([distance for distance, _ in chosen])),
        })
    return pd.DataFrame(rows)


def rank_peers(df: pd.DataFrame, cfg: dict, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = df.loc[peer_eligibility(df, cfg["site_eui_model_max"])].copy()
    parts = [
        compare_group(group, k, cfg["peer_min_neighbors"])
        for _, group in eligible.groupby(["year", "property_type"], observed=True)
    ]
    annual = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    annual["peer_excess_kbtu"] = annual["peer_eui_excess"] * annual["floor_area"]
    summary = annual.groupby("building_id").agg(
        years_with_peers=("year", "nunique"),
        positive_peer_years=("peer_eui_excess", lambda x: int((x > 0).sum())),
        mean_peer_excess_kbtu=("peer_excess_kbtu", "mean"),
        mean_actual_site_eui=("site_eui", "mean"),
        mean_peer_median_site_eui=("peer_median_site_eui", "mean"),
        mean_actual_to_peer_median_ratio=("actual_to_peer_median_ratio", "mean"),
        mean_peer_distance=("peer_distance_mean", "mean"),
        property_type=("property_type", "first"),
        floor_area=("floor_area", "median"),
    ).reset_index()
    ranked = summary[
        (summary["years_with_peers"] == len(cfg["years"]))
        & (summary["positive_peer_years"] >= 2)
        & (summary["mean_peer_excess_kbtu"] > 0)
    ].sort_values("mean_peer_excess_kbtu", ascending=False).reset_index(drop=True)
    ranked["check_data_first"] = ranked["mean_actual_to_peer_median_ratio"].gt(5)
    ranked.insert(0, "peer_priority_rank", range(1, len(ranked) + 1))
    return annual, ranked


def sensitivity_outputs(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankings, top_sets, summary_rows = [], {}, []
    for k in cfg["peer_neighbors_sensitivity"]:
        _, ranked = rank_peers(df, cfg, k)
        n_top = max(1, int(np.ceil(len(ranked) * .10))) if len(ranked) else 0
        top_sets[k] = set(ranked.head(n_top)["building_id"])
        rankings.append(ranked.assign(peer_neighbors=k))
        summary_rows.append({
            "row_type": "ranking", "peer_neighbors": k, "candidate_buildings": len(ranked),
            "top_decile_size": n_top, "comparison_peer_neighbors": pd.NA,
            "top_decile_jaccard": pd.NA,
        })
    for i, first_k in enumerate(cfg["peer_neighbors_sensitivity"]):
        for second_k in cfg["peer_neighbors_sensitivity"][i + 1:]:
            union = top_sets[first_k] | top_sets[second_k]
            summary_rows.append({
                "row_type": "top_decile_overlap", "peer_neighbors": first_k,
                "candidate_buildings": pd.NA, "top_decile_size": pd.NA,
                "comparison_peer_neighbors": second_k,
                "top_decile_jaccard": len(top_sets[first_k] & top_sets[second_k]) / len(union) if union else np.nan,
            })
    return pd.concat(rankings, ignore_index=True), pd.DataFrame(summary_rows)


def main():
    cfg = load_config()
    df = pd.read_csv(ROOT / "data/processed/model_table.csv", dtype={"building_id": str})
    df["log_floor_area"] = np.log(df["floor_area"])
    annual, ranked = rank_peers(df, cfg, cfg["peer_neighbors"])
    context = df[["building_id", "year", "address", "occupancy", "building_age", "number_of_buildings"]]
    annual = annual.merge(context, on=["building_id", "year"], how="left", validate="one_to_one")
    annual.to_csv(ROOT / "outputs/peer_comparisons_annual.csv", index=False)
    ranked.to_csv(ROOT / "outputs/audit_peer_ranking.csv", index=False)

    ranked_context = ranked[["building_id", "peer_priority_rank", "check_data_first"]]
    annual.merge(ranked_context[ranked_context["check_data_first"]], on="building_id", how="inner").sort_values(
        ["peer_priority_rank", "year"]
    ).to_csv(ROOT / "outputs/peer_outlier_review.csv", index=False)

    sensitivity, sensitivity_summary = sensitivity_outputs(df, cfg)
    sensitivity.to_csv(ROOT / "outputs/peer_ranking_sensitivity.csv", index=False)
    sensitivity_summary.to_csv(ROOT / "outputs/peer_sensitivity_summary.csv", index=False)
    print(f"Compared {len(annual):,} building-years and ranked {len(ranked):,} stable peer candidates")


if __name__ == "__main__":
    main()
