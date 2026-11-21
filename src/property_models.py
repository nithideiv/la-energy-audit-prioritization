"""Building-disjoint model selection and property-type routing.

No target-derived variable is a predictor. Models are kept frozen after fitting
on training buildings; validation chooses models, calibration sets intervals,
and test buildings are used only for the final evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["log_floor_area", "building_age", "latitude", "longitude", "year",
           "mean_temp_f", "total_prcp_mm", "hdd65", "cdd65", "hot_days_90f", "occupancy"]
FEATURES = NUMERIC + ["property_type"]
MODEL_NAMES = ["property_type_median", "ridge", "random_forest", "absolute_boosting"]
LABELS = {"property_type_median": "Property-type median", "ridge": "Ridge regression",
          "random_forest": "Random forest", "absolute_boosting": "MAE gradient boosting"}
BLOCKS = {"Floor area (log)": ["log_floor_area"], "Building age": ["building_age"],
          "Location": ["latitude", "longitude"], "Reporting year": ["year"],
          "Annual weather": ["mean_temp_f", "total_prcp_mm", "hdd65", "cdd65", "hot_days_90f"],
          "Occupancy": ["occupancy"], "Property type": ["property_type"]}


def assign_splits(df, seed):
    """Approximately 50/20/10/20% train/validation/calibration/test buildings."""
    def split(frame, fraction, offset):
        s = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset)
        a, b = next(s.split(frame, groups=frame.building_id))
        return frame.iloc[a], frame.iloc[b]
    dev, test = split(df, .2, 0)
    dev, calibration = split(dev, .125, 1)
    train, validation = split(dev, 2 / 7, 2)
    roles = {}
    for name, part in [("train", train), ("validation", validation),
                       ("calibration", calibration), ("test", test)]:
        roles.update(dict.fromkeys(part.building_id.unique(), name))
    return df.building_id.map(roles)


def cohort_masks(df, common_min):
    """Common types are defined with training counts only, before scoring."""
    counts = df.loc[df.split.eq("train")].groupby("property_type").building_id.nunique()
    common = counts[counts.ge(common_min)].index
    return {
        "all_eligible": pd.Series(True, index=df.index),
        "common_types": df.property_type.isin(common),
        "stable_single_building": (~df.needs_data_review.fillna(True)
                                   & df.number_of_buildings.eq(1)
                                   & df.floor_area.ge(1000)),
    }


class TypeMedian:
    def fit(self, x, y):
        work = x[["property_type"]].assign(_target=np.asarray(y))
        self.medians = work.groupby("property_type")._target.median()
        self.fallback = float(np.median(y))
        return self

    def predict(self, x):
        return x.property_type.map(self.medians).fillna(self.fallback).to_numpy(float)


def fit_model(name, train, seed, trees):
    if name == "property_type_median":
        return TypeMedian().fit(train[FEATURES], train.site_eui)
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                       keep_empty_features=True)),
                              ("scale", StandardScaler())]), NUMERIC),
        ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["property_type"]),
    ])
    models = {
        "ridge": lambda: Ridge(alpha=10),
        "random_forest": lambda: RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                                        n_jobs=4, random_state=seed),
        "absolute_boosting": lambda: HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=150, max_leaf_nodes=15, min_samples_leaf=20,
            l2_regularization=1, early_stopping=False, random_state=seed),
    }
    model = Pipeline([("preprocess", prep), ("regressor", models[name]())])
    model.fit(train[FEATURES], train.site_eui)
    return model


def predict(model, frame):
    return np.maximum(0, model.predict(frame[FEATURES]))


def route_predict(frame, general, specialists, choices):
    """Apply the frozen validation-selected route, falling back for unseen types."""
    result = pd.Series(predict(general, frame), index=frame.index)
    for typ, name in choices.items():
        if name == "general":
            continue
        mask = frame.property_type.eq(typ)
        if mask.any():
            result.loc[mask] = predict(specialists[typ][name], frame.loc[mask])
    return result.to_numpy()


def conformal_radius(calibration, prediction, coverage=.9):
    """Cluster conformal: score is a building's maximum annual absolute error.

    With exchangeable buildings, this targets simultaneous coverage of the
    recorded years of a new building. It is marginal across property types.
    """
    scores = (calibration.site_eui - prediction).abs().groupby(calibration.building_id).max()
    rank = int(np.ceil((len(scores) + 1) * coverage))
    if rank > len(scores):
        return float("inf")
    return float(np.sort(scores.to_numpy())[rank - 1])


def cluster_interval(frame, values, seed, n_boot=500):
    """Building bootstrap of a row-weighted mean, retaining all recorded years."""
    g = pd.DataFrame({"id": frame.building_id.to_numpy(), "v": np.asarray(values)}).groupby("id").v.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(g), len(g))
        estimates.append(g["sum"].to_numpy()[ix].sum() / g["count"].to_numpy()[ix].sum())
    return np.quantile(estimates, [.025, .975])
