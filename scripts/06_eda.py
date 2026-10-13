from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "data/processed/model_table.csv")
    out = ROOT / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x="year", y="site_eui", showfliers=False); plt.title("Site EUI by year"); plt.tight_layout(); plt.savefig(out / "site_eui_by_year.png", dpi=180); plt.close()
    top_types = df["property_type"].value_counts().head(10).index
    plt.figure(figsize=(10, 6)); sns.boxplot(data=df[df["property_type"].isin(top_types)], y="property_type", x="site_eui", showfliers=False); plt.title("Site EUI for most common property types"); plt.tight_layout(); plt.savefig(out / "site_eui_by_property_type.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 5)); sns.scatterplot(data=df, x="cdd65", y="site_eui", hue="property_type", legend=False, alpha=.25); plt.title("Cooling degree days and Site EUI"); plt.tight_layout(); plt.savefig(out / "cdd_vs_site_eui.png", dpi=180); plt.close()
    df.isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(ROOT / "outputs/missingness.csv")


if __name__ == "__main__":
    main()

