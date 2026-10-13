from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config


def main():
    cfg = load_config()
    df = pd.read_csv(ROOT / "data/processed/model_table.csv", dtype={"building_id": str})
    df = df[
        df["station_id"].notna()
        & df["mean_temp_f"].notna()
        & df["site_eui"].le(cfg["site_eui_model_max"])
    ].reset_index(drop=True)
    num = ["floor_area", "building_age", "site_eui", "emissions", "hdd65", "cdd65", "hot_days_90f"]
    cat = ["property_type"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    x = prep.fit_transform(df[num + cat])
    scores, fitted = [], {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=461, n_init=20).fit(x)
        labels = model.labels_
        score = silhouette_score(x, labels, sample_size=min(5000, len(df)), random_state=461) if len(set(labels)) > 1 else float("nan")
        scores.append({"k": k, "inertia": model.inertia_, "silhouette": score})
        fitted[k] = model
    score_df = pd.DataFrame(scores)
    score_df.to_csv(ROOT / "outputs/cluster_selection.csv", index=False)
    best_k = int(score_df.loc[score_df["silhouette"].idxmax(), "k"])
    df["cluster"] = fitted[best_k].labels_
    df.to_csv(ROOT / "outputs/building_year_clusters.csv", index=False)
    summary = df.groupby("cluster").agg(rows=("building_id", "size"), buildings=("building_id", "nunique"), median_site_eui=("site_eui", "median"), median_floor_area=("floor_area", "median"), mean_emissions=("emissions", "mean"), mean_hot_days=("hot_days_90f", "mean"))
    summary.to_csv(ROOT / "outputs/cluster_summary.csv")
    print(f"Selected k={best_k} by silhouette score")


if __name__ == "__main__":
    main()
