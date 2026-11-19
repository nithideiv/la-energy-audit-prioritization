import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from project import haversine_miles, normalize_columns


def test_normalize_columns():
    df = pd.DataFrame(columns=["Site EUI (kBtu/ft²)", "BUILDING ADDRESS"])
    assert normalize_columns(df).columns.tolist() == ["site_eui_kbtu_ft", "building_address"]


def test_haversine_zero():
    assert np.isclose(haversine_miles(34, -118, 34, -118), 0)



def _load_cleaner():
    import importlib.util
    spec = importlib.util.spec_from_file_location("clean_buildings", ROOT / "scripts" / "03_clean_buildings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_keeps_valid_occupancy_and_blanks_out_of_range():
    raw = pd.DataFrame({
        "BUILDING ID": ["1", "2", "3", "4"],
        "BUILDING ADDRESS": ["A", "B", "C", "D"],
        "PROGRAM YEAR": [2023] * 4,
        "PROPERTY TYPE": ["Office"] * 4,
        "GROSS BUILDING FLOOR AREA (ft²)": [10000, 20000, 30000, 40000],
        "SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": [50.0, 60.0, 70.0, 80.0],
        "YEAR BUILT": [1990] * 4,
        "OCCUPANCY": [100, 0, 150, None],
    })
    usable, _ = _load_cleaner().clean(raw, [2023])
    occ = usable.set_index("building_id")["occupancy"]
    assert occ["1"] == 100 and occ["2"] == 0   # 0% (vacant) is valid, not missing
    assert np.isnan(occ["3"]) and np.isnan(occ["4"])  # out-of-range and blank both become missing


def test_clean_records_specific_exclusion_reasons():
    raw = pd.DataFrame({
        "BUILDING ID": ["1", "2"], "BUILDING ADDRESS": ["A", "B"],
        "PROGRAM YEAR": [2023, 2023], "PROPERTY TYPE": ["Office", "Office"],
        "GROSS BUILDING FLOOR AREA (ft²)": [None, 10000],
        "SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": [None, None],
        "YEAR BUILT": [1990, 1990],
    })
    _, excluded, _ = _load_cleaner().clean_with_exclusions(raw, [2023])
    assert excluded.loc[excluded["building_id"] == "1", "exclusion_reason"].item() == "missing_or_invalid_floor_area;missing_or_invalid_site_eui"
    assert excluded.loc[excluded["building_id"] == "2", "exclusion_reason"].item() == "missing_or_invalid_site_eui"


def test_clean_reports_exclusions_by_year_and_status_after_dedup():
    raw = pd.DataFrame({
        "BUILDING ID": ["1", "2", "3", "3", "4", "5", "6", "7", "8"],
        "BUILDING ADDRESS": ["A", "B", "C", "C", "D", "E", "F", "G", "H"],
        "PROGRAM YEAR": [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2025, 2022],
        "PROPERTY TYPE": ["Office"] * 9,
        "GROSS BUILDING FLOOR AREA (ft²)": [None, 10000, None, 10000, 10000, 0, 10000, 10000, 10000],
        "SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": [None, None, None, 40, 0, 50, 60, 70, None],
        "YEAR BUILT": [1990] * 9,
        "COMPLIANCE STATUS": [
            " not complied ", "complied", "NOT COMPLIED", "COMPLIED", "NOT COMPLIED",
            "COMPLIED", "COMPLIED", "COMPLIED", "NOT COMPLIED",
        ],
    })

    usable, excluded, report = _load_cleaner().clean_with_exclusions(raw, [2023, 2024, 2025])

    assert report["excluded_rows_by_year_and_compliance_status"] == {
        2023: {"COMPLIED": 1, "NOT COMPLIED": 1},
        2024: {"COMPLIED": 1, "NOT COMPLIED": 1},
    }
    assert len(excluded) == 4
    assert set(usable["building_id"]) == {"3", "6", "7"}
    assert not any(column.startswith("_") for column in usable.columns)
    assert not any(column.startswith("_") for column in excluded.columns)
    assert "_compliance_status" not in report["missing_by_column"]


@pytest.mark.parametrize("statuses", [None, [None, " \t"]])
def test_clean_reports_unknown_for_missing_compliance_status(statuses):
    raw = pd.DataFrame({
        "BUILDING ID": ["1", "2"], "BUILDING ADDRESS": ["A", "B"],
        "PROGRAM YEAR": [2023, 2023], "PROPERTY TYPE": ["Office", "Office"],
        "GROSS BUILDING FLOOR AREA (ft²)": [None, 10000],
        "SITE ENERGY USE INTENSITY (EUI) (kBtu/ft²)": [None, None],
        "YEAR BUILT": [1990, 1990],
    })
    if statuses is not None:
        raw["COMPLIANCE STATUS"] = statuses

    _, _, report = _load_cleaner().clean_with_exclusions(raw, [2023])

    assert report["excluded_rows_by_year_and_compliance_status"] == {2023: {"UNKNOWN": 2}}
