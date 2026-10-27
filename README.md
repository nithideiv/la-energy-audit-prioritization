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
#   Address file: census_batch_input.csv
#   Benchmark:    Public_AR_Current
#   Vintage:      Current_Current

# Save its result as data/raw/census_batch_result.csv, then run:
python scripts/04_geocode_buildings.py --parse data/raw/census_batch_result.csv
python scripts/05_build_model_table.py
python scripts/06_eda.py
python scripts/07_model_and_rank.py      # slowest step: several minutes (random forests, 5-fold + held-out)
python scripts/08_cluster.py
python scripts/09_report_figures.py      # report figures + report_stats.json (seconds; trains nothing)
```

The Census batch geocoder is deliberately a manual checkpoint so no addresses are sent without a team member reviewing the file. If the LA download already includes reliable latitude/longitude, Step 4 will use those and may not need a batch upload. Geocodes accumulate in `data/interim/geocodes.csv`, so after a fresh download only addresses not yet matched are prepared for upload.

### What each step reads and writes

| Step | Script | Main inputs | Main outputs |
|---|---|---|---|
| 1 | `01_download_buildings.py` | LA open-data portal | `data/raw/ebewe_all.csv` |
| 2 | `02_download_weather.py` | NOAA GHCN-Daily, `config.yml` | `data/raw/weather/*.csv`, `data/interim/weather_daily.csv`, `data/processed/weather_annual.csv` |
| 3 | `03_clean_buildings.py` | raw EBEWE | `data/processed/buildings_clean.csv`, `outputs/data_quality_report.json` |
| 4 | `04_geocode_buildings.py` | cleaned buildings, Census batch result | `data/interim/census_batch_input.csv`, `data/interim/geocodes.csv`, `outputs/geocoding_report.json` |
| 5 | `05_build_model_table.py` | cleaned buildings, geocodes, annual weather | `data/processed/model_table.csv`, `outputs/join_report.json` |
| 6 | `06_eda.py` | model table | EDA figures, `outputs/missingness.csv` |
| 7 | `07_model_and_rank.py` | model table | metrics, out-of-fold predictions, ranking, stability, extreme-EUI review file |
| 8 | `08_cluster.py` | model table | cluster tables |
| 9 | `09_report_figures.py` | outputs of steps 3–8 | report figures, `outputs/report_stats.json` |

## 3. Outputs

Data files:

- `data/processed/buildings_clean.csv`: cleaned building-year rows
- `data/processed/weather_annual.csv`: station-year weather features
- `data/processed/model_table.csv`: modeling table after nearest-station join

Reports and tables in `outputs/`:

- `data_quality_report.json`: exclusions, duplicates, coverage, the source column used for the year (`year_source_column`), and usable rows per year (`usable_rows_by_year`)
- `geocoding_report.json`: match rate
- `join_report.json`: rows geocoded and joined to weather, median station distance
- `missingness.csv`: missing fraction per column in the model table
- `model_metrics.csv`: grouped out-of-fold and fixed held-out metrics, including baselines
- `oof_predictions.csv`: predictions made without training on the scored building
- `extreme_site_eui_review.csv`: rows above the Site EUI cap (`site_eui_model_max` in `config.yml`), set aside for review and not modeled
- `audit_priority_ranking.csv`: three-year candidate ranking, with `actual_to_predicted_ratio` and a `check_data_first` flag (ratio above 5) for likely data problems or special uses; the flag does not change the order
- `model_ranking_stability.csv`: top-decile overlap (Jaccard) across models
- `cluster_selection.csv`, `cluster_summary.csv`, `building_year_clusters.csv`: exploratory k-means results
- `report_stats.json`: every number quoted in the report (variable summaries, weather-effect bootstrap, residual persistence, ranking concentration, top 10, plus the metrics and reports above)

Figures in `outputs/figures/`:

- `site_eui_by_year.png`, `site_eui_by_property_type.png`, `cdd_vs_site_eui.png`: exploratory plots (step 6)
- `report_fig1_eui_distribution.png`, `report_fig2_correlations.png`, `report_fig3_model_mae.png`, `report_fig4_residual_persistence.png`, `report_fig5_ranking.png`: report figures (step 9)

### Rerunning with new data

Everything downstream of the download is recomputed from the data, so rerunning steps 1–9 updates all tables and figures. Step 9 computes its labels (counts, medians, flagged totals, best model) from the outputs, so the figures stay consistent with the data. The written report is not regenerated: after a rerun, take new numbers from `outputs/report_stats.json`. The LA portal is updated over time, so a later download can change counts slightly (an earlier download had 21,418 usable rows and the October 26, 2026 download had 21,523).

## 4. Methodological guardrails

- All rows from one building stay in the same fold.
- Imputation, one-hot encoding, scaling, and model fitting occur inside each training fold.
- Energy-derived variables (ENERGY STAR score, weather-normalized EUI, emissions, total energy) are not predictors.
- Rows with Site EUI above `site_eui_model_max` are set aside for review instead of being modeled.
- Buildings are ranked only if they have usable records in all three years and positive residuals in at least two years.
- `excess_kbtu = (actual Site EUI − out-of-fold predicted Site EUI) × floor area`.
- Negative residuals are retained in annual predictions; they are not silently set to zero.
- Rankings include model-choice sensitivity and top-decile overlap, not just one final ordered list.
- Treat `check_data_first` buildings as records to verify before treating them as audit candidates.

## 5. Interpreting the results

Compare models on the grouped out-of-fold rows of `model_metrics.csv` (the held-out rows use a different subset, so the two scopes are not directly comparable). In the October 26, 2026 run:

- A per-property-type median had the lowest MAE (22.2 kBtu/ft²), ahead of linear regression (24.6) and random forest (24.8). Linear regression had the best RMSE and R² (0.14).
- Adding annual weather changed MAE by less than 0.02 kBtu/ft²; bootstrap intervals (in `report_stats.json`) include zero.
- The ranking is therefore best read as a screen for persistent unexplained use, not as a measure of inefficiency. `07_model_and_rank.py` uses the random forest with weather as its primary model, which is not the lowest-MAE model, so check `model_ranking_stability.csv` before relying on exact rank positions.

These numbers will shift if the data change; see `report_stats.json` for the current values.

## 6. Tests

```bash
pytest -q tests
```

## 7. Team split

| Owner | Primary work | Handoff |
|---|---|---|
| Reda | EBEWE download, schema log, building cleaning | `buildings_clean.csv` + quality report |
| Daryl | NOAA download, quality flags, annual aggregation | `weather_annual.csv` |
| Srujan | Address review, Census batch, nearest station join | geocodes + match report |
| Nithi | EDA, grouped models, rankings, stability checks, report figures | figures + metrics + ranking + `report_stats.json` |

Everyone should review the final exclusion rules and interpretation before submission.

## 8. Data sources

- LA EBEWE: https://data.lacity.org/d/9yda-i4ya
- NOAA GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Census Geocoder: https://geocoding.geo.census.gov/geocoder/

## 9. Year definition

The current portal export labels the year field `Program Year` (the portal description calls it the "Compliance Year"). The cleaning script reads it as `year` and also accepts the aliases `Compliance Year` and `Reporting Year`; the column actually used is recorded as `year_source_column` in `outputs/data_quality_report.json`.

**Decision:** `Program Year` N is treated as the calendar year N of energy consumption, not a later filing year, so each building-year is matched to NOAA weather from the same calendar year.

**Evidence:**
- LADBS compliance instructions say utility data must be entered "for the applicable calendar year" matching the program year (e.g., PY 2019, 2020, 2021), and that PY 2019 data cannot be reused for PY 2020 or 2021. The 2018 scope-and-instructions document likewise ties Benchmark Year 2019 to calendar year 2019.
- The data.lacity.org / data.gov dataset description says only that owners file a benchmark report "for the Compliance Year" and does not define it further.

**Caveats for the final report:**
- The evidence comes from LADBS instruction documents for earlier program years. We did not find a statement specific to PY 2023–2025, so confirm with LADBS (ladbs.ebewe@lacity.org) if the exact wording matters.
- Filing for the most recent program year lags the consumption year. PY 2025 (calendar-2025 data, filed in 2026) has fewer usable rows than earlier years (6,603 against 7,566 for 2023 in the October 26, 2026 download), so the three-year ranking eligibility rule may exclude more buildings than it otherwise would.
- The original year field is kept unchanged in `buildings_clean.csv`.
