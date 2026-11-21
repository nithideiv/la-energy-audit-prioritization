import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_reporting import same_type_context, file_sha256, verify_analysis_files


def extras():
    spec = importlib.util.spec_from_file_location("extras_reporting", ROOT / "scripts/12_property_type_extras.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_card_counts_exact_type_instead_of_pooled_fallback():
    screen = pd.DataFrame({"building_id": ["a", "b", "c", "d"],
                           "property_type": ["Refrigerated Warehouse", "Refrigerated Warehouse", "Other", "Other"],
                           "three_year": [True] * 4, "mean_gap_kbtu": [50, 10, 100, 200]})
    c = same_type_context(screen, "a")
    assert c["same_type_three_year_buildings"] == 2
    assert c["other_same_type_buildings"] == 1
    assert c["gap_percentile_strictly_lower"] == 100


def test_card_with_no_exact_type_peers_has_no_percentile():
    screen = pd.DataFrame({"building_id": ["a"], "property_type": ["Hospital"],
                           "three_year": [True], "mean_gap_kbtu": [50]})
    assert same_type_context(screen, "a")["gap_percentile_strictly_lower"] is None


def test_mixed_analysis_files_are_rejected():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        artifact = folder / "heldout_predictions.csv"
        artifact.write_text("original run")
        (folder / "run_manifest.json").write_text(json.dumps({"artifact_sha256": {artifact.name: file_sha256(artifact)}}))
        verify_analysis_files(folder)
        artifact.write_text("different run")
        try:
            verify_analysis_files(folder)
        except ValueError as error:
            assert "Analysis file changed" in str(error)
        else:
            raise AssertionError("Mixed analysis files were accepted")


def test_scaled_conformal_uses_order_statistic_not_interpolation():
    m = extras()
    assert m.conformal_quantile(np.arange(19.), .9) == 17
    assert np.isinf(m.conformal_quantile([1.], .9))


def test_wilson_errorbar_lengths_never_go_negative_at_boundaries():
    m = extras()
    for n in range(1, 201):
        for k in [0, n // 2, n]:
            p, low, high = m.wilson(k, n)
            assert 0 <= low <= p <= high <= 1


def test_limited_test_support_does_not_change_validation_route():
    m = extras()
    rec = pd.DataFrame({"property_type": ["Example"], "train_buildings": [60],
                        "validation_buildings": [20], "test_buildings": [5], "limited_test_support": [True],
                        "general_model": ["absolute_boosting"], "selected_route": ["ridge"],
                        "general_MAE": [10], "specialist_MAE": [9], "portfolio_MAE": [9],
                        "portfolio_minus_general_MAE": [-1], "paired_diff_ci_low": [-2], "paired_diff_ci_high": [1],
                        "portfolio_normalized_MAE_pct": [20], "portfolio_signed_error": [0],
                        "general_MAE_low": [5], "general_MAE_high": [15]})
    held = pd.DataFrame({"property_type": ["Example"], "site_eui": [10], "candidate_property_type_median": [12]})
    output = m.recommendations(held, rec)
    assert output.selected_route.iloc[0] == "ridge"
    assert "Ridge regression" in output.recommendation.iloc[0]
