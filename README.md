# LA Building Energy Audit Prioritization

Project for CSCI 461. It downloads 2023–2025 Los Angeles EBEWE records and NOAA GHCN-Daily observations, cleans and joins them, compares building-only and building-plus-weather regression models using building-grouped validation, and produces a cautious audit-priority ranking.

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
| Reda | EBEWE download, schema log, building cleaning | `buildings_clean.csv` + quality report |
| Daryl | NOAA download, quality flags, annual aggregation | `weather_annual.csv` |
| Srujan | Address review, Census batch, nearest station join | geocodes + match report |
| Nithi | EDA, grouped models, rankings, stability checks | figures + metrics + ranking |

Everyone should review the final exclusion rules and interpretation before submission.

## 6. Data sources

- LA EBEWE: https://data.lacity.org/d/9yda-i4ya
- NOAA GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Census Geocoder: https://geocoding.geo.census.gov/geocoder/

## 7. Year definition 

The current portal export labels the year field `Program Year` (the portal description calls it the "Compliance Year"). The cleaning script reads it as `year` and also accepts the aliases `Compliance Year` and `Reporting Year`; the column actually used is recorded as `year_source_column` in `outputs/data_quality_report.json`.

**Decision:** `Program Year` N is treated as the calendar year N of energy consumption, not a later filing year, so each building-year is matched to NOAA weather from the same calendar year.

**Evidence:**
- LADBS compliance instructions say utility data must be entered "for the applicable calendar year" matching the program year (e.g., PY 2019, 2020, 2021), and that PY 2019 data cannot be reused for PY 2020 or 2021. The 2018 scope-and-instructions document likewise ties Benchmark Year 2019 to calendar year 2019.
- The data.lacity.org / data.gov dataset description says only that owners file a benchmark report "for the Compliance Year" and does not define it further.

**Caveats for the final report:**
- The evidence comes from LADBS instruction documents for earlier program years. We did not find a statement specific to PY 2023–2025, so confirm with LADBS (ladbs.ebewe@lacity.org) if the exact wording matters.
- Filing for the most recent program year lags the consumption year. PY 2025 (calendar-2025 data, filed in 2026) has a lower share of rows with a Site EUI than earlier years (about 58% versus 65–68%), so 2025 may be incomplete and the three-year ranking eligibility rule may exclude more buildings than it otherwise would.
- The original year field is kept unchanged in `buildings_clean.csv`.

