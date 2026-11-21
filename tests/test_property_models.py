"""Checks for leakage boundaries, routing fallbacks, and interval semantics."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from property_models import assign_splits, cohort_masks, conformal_radius, route_predict, TypeMedian, FEATURES


def test_building_years_never_cross_splits():
    frame = pd.DataFrame({"building_id": np.repeat(np.arange(200).astype(str), 3)})
    frame["split"] = assign_splits(frame, 461)
    assert frame.groupby("building_id").split.nunique().eq(1).all()
    assert set(frame.split) == {"train", "validation", "calibration", "test"}
    assert frame.split.equals(assign_splits(frame, 461))


def test_common_type_filter_cannot_learn_from_test_frequency():
    frame = pd.DataFrame({"building_id": ["a", "b", "c", "d"],
                          "split": ["train", "test", "test", "test"],
                          "property_type": ["Office", "Warehouse", "Warehouse", "Warehouse"],
                          "needs_data_review": [False] * 4, "number_of_buildings": [1] * 4,
                          "floor_area": [2000] * 4})
    assert cohort_masks(frame, 1)["common_types"].tolist() == [True, False, False, False]


def test_unseen_type_uses_training_median_only():
    model = TypeMedian().fit(pd.DataFrame({"property_type": ["Office", "Office", "Shop"]}), [10, 30, 100])
    result = model.predict(pd.DataFrame({"property_type": ["Office", "Shop", "Unseen"]}))
    np.testing.assert_allclose(result, [20, 100, 30])


def test_portfolio_falls_back_for_unseen_type_and_preserves_row_order():
    class Constant:
        def __init__(self, value): self.value = value
        def predict(self, x): return np.full(len(x), self.value)
    frame = pd.DataFrame(0., index=[8, 2, 6], columns=FEATURES)
    frame["property_type"] = ["Office", "Unseen", "Shop"]
    p = route_predict(frame, Constant(20), {"Office": {"ridge": Constant(50)}},
                      {"Office": "ridge", "Shop": "general"})
    np.testing.assert_allclose(p, [50, 20, 20])


def test_calibration_scores_building_maximum_and_finite_sample_quantile():
    # Nine buildings: 90% split-conformal rank is ceil(10*.9)=9 (the maximum).
    cal = pd.DataFrame({"building_id": np.repeat(np.arange(9), 3),
                        "site_eui": np.tile([1., 2., 3.], 9)})
    cal.loc[26, "site_eui"] = 99
    assert conformal_radius(cal, np.zeros(len(cal))) == 99


def test_target_derived_variables_are_never_model_features():
    assert set(FEATURES).isdisjoint({"site_eui", "emissions", "excess_kbtu", "predicted_site_eui",
                                   "energy_star_score", "weather_normalized_site_eui"})
