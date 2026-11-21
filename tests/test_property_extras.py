import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("extras", ROOT / "scripts" / "12_property_type_extras.py")
extras = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extras)


def test_wilson_bounds_and_empty():
    p, lo, hi = extras.wilson(9, 10)
    assert 0 <= lo < p < hi <= 1
    assert np.isnan(extras.wilson(0, 0)[0])


def test_scaled_radius_covers_calibration_buildings_at_level():
    rng = np.random.default_rng(0)
    n = 400
    cal = pd.DataFrame({"building_id": np.repeat(np.arange(n), 3), "site_eui": rng.lognormal(3, .5, n * 3)})
    pred = np.full(len(cal), cal["site_eui"].median())
    q = extras.scaled_radius(cal, pred, level=.9)
    low, high = extras.scaled_interval(pred, q)
    inside = cal["site_eui"].between(low, high).groupby(cal["building_id"]).all().mean()
    assert inside >= .9


def test_scaled_interval_floor_and_nonnegative():
    low, high = extras.scaled_interval(np.array([0.0, 100.0]), q=.5, floor=5.0)
    assert low[0] == 0 and np.isclose(high[0], 2.5)
    assert np.isclose(low[1], 50) and np.isclose(high[1], 150)


def test_building_screen_rule_and_rank():
    rows = []
    for b, (resid, area) in {"a": ([5, 5, 5], 1000), "b": ([5, -1, -1], 1000), "c": ([-5, -5, -5], 1000), "d": ([9, 9, 9], 2000)}.items():
        for year, r in zip([2023, 2024, 2025], resid):
            rows.append({"building_id": b, "year": year, "site_eui": 50 + r, "pred": 50.0, "floor_area": area,
                         "property_type": "Office", "above": r > 8})
    rows.append({"building_id": "e", "year": 2023, "site_eui": 80, "pred": 50.0, "floor_area": 1000, "property_type": "Office", "above": True})
    s = extras.building_screen(pd.DataFrame(rows), "pred", "above", 3).set_index("building_id")
    assert list(s.index[s["qualifies"]]) == ["a", "d"]          # b: one positive year; c: negative; e: one year only
    assert s.loc["d", "rank"] == 1 and s.loc["a", "rank"] == 2
    assert s.loc["d", "outside_range"] and not s.loc["a", "outside_range"]
