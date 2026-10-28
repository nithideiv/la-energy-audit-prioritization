import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
