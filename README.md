# LA Building Energy Audit Prioritization

Starter project for CSCI 461. It downloads 2023–2025 Los Angeles EBEWE records and NOAA GHCN-Daily observations, cleans and joins them, compares building-only and building-plus-weather regression models using building-grouped validation, and produces a cautious audit-priority ranking.

## Research question

Can building characteristics, location, reporting year, and annual weather predict Site Energy Use Intensity (Site EUI), and which buildings show repeated positive out-of-fold residuals during 2023–2025?

The ranking is a screening tool—not proof of inefficiency or guaranteed retrofit savings. Large residuals can reflect occupancy, operating hours, unusual uses, or reporting errors.

## 1. Setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run order

```bash
python scripts/01_download_buildings.py
python scripts/02_download_weather.py
python scripts/03_clean_buildings.py
python scripts/04_geocode_buildings.py --prepare
# Upload data/interim/census_batch_input.csv at the Census batch geocoder. Here's the set up:
#Address file: census_batch_input.csv
#Benchmark: Public_AR_Current
#Vintage: Current_Current


# Save its result as data/raw/census_batch_result.csv, then run:
python scripts/04_geocode_buildings.py --parse data/raw/census_batch_result.csv
python scripts/05_build_model_table.py
python scripts/06_eda.py
python scripts/07_model_and_rank.py
python scripts/08_cluster.py
```

The Census batch geocoder is deliberately a manual checkpoint so no addresses are sent without a team member reviewing the file. If the LA download already includes reliable latitude/longitude, Step 4 will use those and may not need a batch upload.

## 3. Outputs

- `data/processed/buildings_clean.csv`: cleaned building-year rows
- `data/processed/weather_annual.csv`: station-year weather features
- `data/processed/model_table.csv`: modeling table after nearest-station join
- `outputs/data_quality_report.json`: exclusions, duplicates, and coverage
- `outputs/geocoding_report.json`: match rate and station-distance summary
- `outputs/model_metrics.csv`: grouped held-out metrics
- `outputs/oof_predictions.csv`: predictions made without training on the scored building
- `outputs/audit_priority_ranking.csv`: three-year candidate ranking
- `outputs/model_ranking_stability.csv`: top-decile overlap across models
- `outputs/cluster_summary.csv`: exploratory cluster descriptions
- `outputs/figures/`: exploratory and evaluation plots

## 4. Methodological guardrails

- All rows from one building stay in the same fold.
- Imputation, one-hot encoding, scaling, and model fitting occur inside each training fold.
- Energy-derived variables (ENERGY STAR score, weather-normalized EUI, emissions, total energy) are not predictors.
- Buildings are ranked only if they have usable records in all three years and positive residuals in at least two years.
- `excess_kbtu = (actual Site EUI − out-of-fold predicted Site EUI) × floor area`.
- Negative residuals are retained in annual predictions; they are not silently set to zero.
- Rankings include model-choice sensitivity and top-decile overlap, not just one final ordered list.

## 5. Team split

| Owner | Primary work | Handoff |
|---|---|---|
| Person 1 | EBEWE download, schema log, building cleaning | `buildings_clean.csv` + quality report |
| Person 2 | NOAA download, quality flags, annual aggregation | `weather_annual.csv` |
| Person 3 | Address review, Census batch, nearest station join | geocodes + match report |
| Person 4 | EDA, grouped models, rankings, stability checks | figures + metrics + ranking |

Everyone should review the final exclusion rules and interpretation before submission.

## 6. Data sources

- LA EBEWE: https://data.lacity.org/d/9yda-i4ya
- NOAA GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Census Geocoder: https://geocoding.geo.census.gov/geocoder/

## 7. Important issue to verify

Confirm whether the EBEWE `Compliance Year` corresponds to the calendar year of energy consumption or to a later filing year. Keep the original year field and document the decision in the final report. The pipeline assumes the selected year is matched to weather from the same calendar year, but this should not be treated as settled until verified from the dataset documentation.

The current portal export labels this field `Program Year`; the cleaning script also accepts older aliases such as `Compliance Year` and `Reporting Year`.

