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

