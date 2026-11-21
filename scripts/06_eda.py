from pathlib import Path
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config
from audit_reporting import file_sha256


def main():
    df = pd.read_csv(ROOT / "data/processed/model_table.csv")
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x="year", y="site_eui", showfliers=False); plt.title("Site EUI by year"); plt.tight_layout(); plt.savefig(out / "site_eui_by_year.png", dpi=180); plt.close()
    top_types = df["property_type"].value_counts().head(10).index
    plt.figure(figsize=(10, 6)); sns.boxplot(data=df[df["property_type"].isin(top_types)], y="property_type", x="site_eui", showfliers=False); plt.title("Site EUI for most common property types"); plt.tight_layout(); plt.savefig(out / "site_eui_by_property_type.png", dpi=180); plt.close()
    cap = load_config()["site_eui_model_max"]
    valid = (df["site_eui"].gt(0) & df["site_eui"].le(cap) & df["floor_area"].gt(0)
             & df["station_id"].notna() & df["mean_temp_f"].notna())
    modeled = df.loc[valid]
    # Extreme records remain available for review instead of setting the plot's scale.
    df.loc[df["site_eui"].gt(cap)].to_csv(ROOT / "outputs/eda_extreme_site_eui_review.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=modeled, x="cdd65", y="site_eui", color="#2b6cb0",
                    legend=False, alpha=.12, s=12, ax=ax)
    ax.set_yscale("log")
    ax.set(xlabel="Cooling degree days (base 65°F)", ylabel="Site EUI (kBtu/ft²/year, log scale)",
           title=f"Cooling degree days and Site EUI · {len(modeled):,} eligible building-years")
    fig.text(.02, -.025, f"Joined modeling sample; positive EUI ≤ {cap:,}. Extreme records are in eda_extreme_site_eui_review.csv. Weather repeats within station/year.", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cdd_vs_site_eui.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    df.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(ROOT / "outputs/missingness.csv")
    (ROOT / "outputs/eda_manifest.json").write_text(json.dumps({
        "input_sha256": file_sha256(ROOT / "data/processed/model_table.csv"),
        "cdd_figure_sha256": file_sha256(out / "cdd_vs_site_eui.png"),
        "eligible_scatter_rows": len(modeled), "eui_review_cap": cap}, indent=2))


if __name__ == "__main__":
    main()

