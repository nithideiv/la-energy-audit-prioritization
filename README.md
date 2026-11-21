# LA Building Energy Audit Prioritization

Project downloads 2023–2025 Los Angeles EBEWE records and NOAA GHCN-Daily observations, cleans and joins them, compares building-only and building-plus-weather regression models using building-grouped validation, and produces a cautious audit-priority ranking. A second analysis (steps 11–12) asks whether per-property-type models beat one general model, recommends which model to use for which property type, and reports feature importance and fairness by property type.

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
python scripts/07_model_and_rank.py      # slowest of the first ten: several minutes (random forests, 5-fold + held-out)
python scripts/08_cluster.py
python scripts/10_peer_compare.py        # explicit like-for-like building comparisons
python scripts/09_report_figures.py      # report figures + report_stats.json (seconds; trains nothing)
python scripts/11_property_type_results.py   # per-type models, model portfolio, held-out test, importance, HTML report
python scripts/12_property_type_extras.py    # recommendation table, fairness, interval coverage, building audit cards
pytest -q tests
```

Step 12 options:

```bash
python scripts/12_property_type_extras.py --cohort stable_single_building   # default; other cohorts: all_eligible, common_types
python scripts/12_property_type_extras.py --building-id 10353               # card for a specific building (repeatable)
python scripts/12_property_type_extras.py --interval scaled                 # type-scaled interval radius instead of constant
python scripts/12_property_type_extras.py --no-report                       # do not insert the extras into the HTML report
```

Step 11 reads the `property_type_analysis` block of `config.yml` (forest size, minimum buildings for a specialist model, minimum validation gain, and so on). Step 12 reads only what step 11 wrote, so rerun 11 first if 12 reports a missing file.

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
| 9 | `10_peer_compare.py` | model table | annual peer comparisons and peer-based audit ranking |
| 10 | `09_report_figures.py` | outputs of steps 3–8 plus peer ranking | report figures, `outputs/report_stats.json` |
| 11 | `11_property_type_results.py` | model table (needs `number_of_buildings` and `needs_data_review` from step 5), `config.yml` | `outputs/property_type_results/`: held-out predictions, per-type test metrics, calibration predictions, feature-block importance, HTML report |
| 12 | `12_property_type_extras.py` | outputs of step 11 | recommendation table, interval coverage, selection fairness, held-out audit screen, figures 08–12, building audit cards |

## 3. Outputs

Data files:

- `data/processed/buildings_clean.csv`: cleaned building-year rows
- `data/processed/buildings_excluded.csv`: excluded building-years with a specific exclusion reason
- `data/processed/weather_annual.csv`: station-year weather features
- `data/processed/model_table.csv`: modeling table after nearest-station join

Reports and tables in `outputs/`:

- `data_quality_report.json`: exclusions, duplicates, coverage, the source column used for the year (`year_source_column`), and usable rows per year (`usable_rows_by_year`).
  `excluded_rows_by_year_and_compliance_status` counts excluded records by year and compliance status after duplicate removal.
  Each excluded record is counted once, even if both floor area and Site EUI are invalid.
  Status labels are trimmed and upper-cased. Missing or blank statuses (including a missing source column) use `UNKNOWN`.
  Only year/status groups with exclusions are included.
- `geocoding_report.json`: match rate
- `join_report.json`: rows geocoded and joined to weather, median station distance
- `missingness.csv`: missing fraction per column in the model table
- `model_metrics.csv`: grouped out-of-fold and fixed held-out metrics, including baselines
- `oof_predictions.csv`: predictions made without training on the scored building
- `extreme_site_eui_review.csv`: rows above the Site EUI cap (`site_eui_model_max` in `config.yml`), set aside for review and not modeled
- `audit_priority_ranking.csv`: three-year candidate ranking, with `actual_to_predicted_ratio` and a `check_data_first` flag (ratio above 5) for likely data problems or special uses; the flag does not change the order
- `building_consistency_flags.csv`, `building_data_review_queue.csv`: buildings whose reported area, year built, property type, or number of buildings need review, with the reported values for each year.
- `peer_comparisons_annual.csv`, `audit_peer_ranking.csv`: direct comparisons with the 10 nearest same-type, same-year peers, matched on size, age, location, and occupancy. Only stable one-building records with at least 8 peers are compared. `check_data_first` flags an average actual-to-peer-median ratio above 5.
- `peer_outlier_review.csv`: annual peer comparisons for every `check_data_first` candidate, to review before recommending an audit.
- `peer_ranking_sensitivity.csv`, `peer_sensitivity_summary.csv`: rankings for 8, 10, and 15 peers and the overlap of their top deciles.
- `model_ranking_stability.csv`: top-decile overlap (Jaccard) across models
- `cluster_selection.csv`, `cluster_summary.csv`, `building_year_clusters.csv`: exploratory k-means results
- `report_stats.json`: every number quoted in the report (variable summaries, weather-effect bootstrap, residual persistence, ranking concentration, top 10, plus the metrics and reports above)

Figures in `outputs/figures/`:

- `site_eui_by_year.png`, `site_eui_by_property_type.png`, `cdd_vs_site_eui.png`: exploratory plots (step 6)
- `report_fig1_eui_distribution.png`, `report_fig2_correlations.png`, `report_fig3_model_mae.png`, `report_fig4_residual_persistence.png`, `report_fig5_ranking.png`: report figures (step 9)

### Per-property-type analysis (steps 11–12), in `outputs/property_type_results/`

Question: does fitting models for subsets of building types predict better than one general model, and which model should be used for which property type?

**Design.** Buildings are split into train / validation / calibration / test (50 / 20 / 10 / 20 %), with all years of a building in the same split. Models are frozen after the validation step, and the test set never influences which model is chosen for a type. The analysis is repeated on three cohorts (different ways to filter the data):

- `all_eligible`: every building that passes the basic modeling filters
- `common_types`: only property types with at least 100 training buildings
- `stable_single_building` (primary): one-building reporting boundary, floor area at least 1,000 ft², and no `needs_data_review` flag

**Candidate models.** Property-type median, ridge, random forest, and absolute-error gradient boosting. The best general model is selected on validation data.

**Thin buckets.** A property type is routed to its own specialist model only if it has at least 60 training and 20 validation buildings and the specialist improves validation MAE by at least 5% over the selected general model. Otherwise the type uses the general model. Types below `min_report_test_buildings` (20) test buildings are not reported separately.

**Intervals.** 90% split-conformal intervals from the calibration buildings (the largest absolute annual error per building), with a constant radius by default or a type-scaled radius with `--interval scaled`. Coverage varies by type, which is reported in `interval_coverage_by_type.csv` rather than hidden.

**Main files** (step 11):

- `heldout_predictions.csv`: general and portfolio prediction for every test building-year
- `per_type_test_metrics.csv`: per-type error for the general model, the portfolio, and the type median
- `calibration_predictions_<cohort>.csv`: calibration-split predictions used by step 12 for intervals
- `feature_block_importance.csv`: permutation importance by feature block and, for step 12, single columns and within-type scopes
- the HTML results report

**Main files** (step 12):

- `recommendations_by_type.csv`, `recommendations.md`: for each property type, which model to use and why (the takeaway table)
- `interval_coverage_by_type.csv`: observed coverage of the 90% interval by type
- `selection_fairness_by_type.csv`: error and audit-selection rate by property type
- `heldout_screen.csv`: audit screen applied to held-out buildings only
- Figures: 08 recommendation by type, 09 fairness by type, 10 interval coverage, 11 actual vs. predicted, 12 feature importance
- `figures/building_cards/<building_id>_audit_card.png`: the view for an individual building manager (reported vs. expected energy per year, typical error band, and why the building does or does not meet the audit screen)

Limitations to state when reporting these results: the analysis uses one split, so a specialist route that wins by a small margin is provisional; the common-types cohort depends on which buildings land in training; the audit ranking from step 12 covers held-out buildings only; interval coverage is uneven across property types.

### Rerunning with new data

Everything downstream of the download is recomputed from the data, so rerunning steps 1–12 updates all tables and figures. Step 9 computes its labels (counts, medians, flagged totals, best model) from the outputs, so the figures stay consistent with the data. The written report is not regenerated: after a rerun, take new numbers from `outputs/report_stats.json` and `outputs/property_type_results/`. The LA portal is updated over time, so a later download can change counts slightly (an earlier download had 21,418 usable rows and the October 26, 2026 download had 21,523).

## 4. Methodological guardrails

- All rows from one building stay in the same fold or split.
- Imputation, one-hot encoding, scaling, and model fitting occur inside each training fold.
- Energy-derived variables (ENERGY STAR score, weather-normalized EUI, emissions, total energy) are not predictors.
- Rows with Site EUI above `site_eui_model_max` are set aside for review instead of being modeled.
- Buildings are ranked only if they have usable records in all three years and positive residuals in at least two years.
- `excess_kbtu = (actual Site EUI − predicted Site EUI) × floor area`.
- Negative residuals are retained in annual predictions; they are not silently set to zero.
- Rankings include model-choice sensitivity and top-decile overlap, not just one final ordered list.
- Treat `check_data_first` buildings as records to verify before treating them as audit candidates.
- The peer ranking excludes records with changing reported core characteristics, multiple-building reporting boundaries, or floor area below 1,000 ft². It ranks only buildings above their peer median in at least two years and on average.
- Occupancy (`OCCUPANCY` in the raw file, a self-reported 0-100 percent) is tested only in the `*_occ` models, which are scored on the same folds and seed as their non-occupancy counterparts. It is blank only for buildings that never filed energy data, so it is missing in 0% of modeled rows. Values outside 0-100 are treated as missing. The ranking still uses `rf_weather`.
- In steps 11–12, the test split never alters which model is routed to a property type, and a property type is never fit on its own below the training and validation thresholds in `config.yml`.

## 5. Interpreting the results

Compare models on the grouped out-of-fold rows of `model_metrics.csv` (the held-out rows use a different subset, so the two scopes are not directly comparable). In the October 26, 2026 run:

- A per-property-type median had the lowest MAE (22.2 kBtu/ft²), ahead of linear regression (24.6) and random forest (24.8). Linear regression had the best RMSE and R² (0.14).
- Adding annual weather changed MAE by less than 0.02 kBtu/ft²; bootstrap intervals (in `report_stats.json`) include zero.
- The ranking is therefore best read as a screen for persistent unexplained use, not as a measure of inefficiency. `07_model_and_rank.py` uses the random forest with weather as its primary model, which is not the lowest-MAE model, so check `model_ranking_stability.csv` before relying on exact rank positions.

- Adding occupancy (`linear_weather_occ`, `rf_weather_occ`) lowered grouped out-of-fold MAE by 0.03 (linear) and 0.06 (random forest) kBtu/ft², and the bootstrap intervals in `occupancy_effect_bootstrap` include zero, so the gain is not distinguishable from noise. Random forest RMSE and R² were slightly worse with it. Occupancy is 95-100% for most buildings, so it carries little signal; the exception is the few buildings below 50% occupied, which have much lower EUI.

Per-property-type results (steps 11–12, stable single-building cohort, test split, October 2026 run). These come from a different split than `model_metrics.csv`, so the MAE values are not comparable with the 22.2 / 24.6 / 24.8 above:

- The selected general model is absolute-error gradient boosting with test MAE 19.42 kBtu/ft². The portfolio (general model plus routed types) has test MAE 19.39, a difference of −0.03 with a 95% interval of [−0.07, 0.01], so the portfolio is not distinguishable from the general model overall.
- Both are better than the property-type median baseline (20.53); the general-minus-baseline interval is [−1.63, −0.61].
- Only Distribution Center and Self-Storage were routed away from the general model (to the property-type median). Every other type uses the general model. That is the takeaway: for most property types one general model is enough, and for those two the simple median is as good or better.
- Property type dominates feature importance; annual weather adds about 1.5% of the model's MAE when the weather block is shuffled.
- 90% interval coverage on test buildings is about 91% overall but uneven across property types.

These numbers will shift if the data change; see `report_stats.json` and `outputs/property_type_results/` for the current values.

## 6. Tests

```bash
pytest -q tests
```

This includes `tests/test_property_extras.py`, which checks the Wilson interval, the type-scaled conformal radius, and the audit-screen rule and ranking used in step 12.

## 7. Data sources

- LA EBEWE: https://data.lacity.org/d/9yda-i4ya
- NOAA GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- Census Geocoder: https://geocoding.geo.census.gov/geocoder/

## 8. Year definition

The current portal export labels the year field `Program Year` (the portal description calls it the "Compliance Year"). The cleaning script reads it as `year` and also accepts the aliases `Compliance Year` and `Reporting Year`; the column actually used is recorded as `year_source_column` in `outputs/data_quality_report.json`.

**Decision:** `Program Year` N is treated as the calendar year N of energy consumption, not a later filing year, so each building-year is matched to NOAA weather from the same calendar year.

**Evidence:**
- LADBS compliance instructions say utility data must be entered "for the applicable calendar year" matching the program year (e.g., PY 2019, 2020, 2021), and that PY 2019 data cannot be reused for PY 2020 or 2021. The 2018 scope-and-instructions document likewise ties Benchmark Year 2019 to calendar year 2019.
- The data.lacity.org / data.gov dataset description says only that owners file a benchmark report "for the Compliance Year" and does not define it further.

**Caveats for the final report:**
- The evidence comes from LADBS instruction documents for earlier program years. We did not find a statement specific to PY 2023–2025, so confirm with LADBS (ladbs.ebewe@lacity.org) if the exact wording matters.
- Filing for the most recent program year lags the consumption year. PY 2025 (calendar-2025 data, filed in 2026) has fewer usable rows than earlier years (6,603 against 7,566 for 2023 in the October 26, 2026 download), so the three-year ranking eligibility rule may exclude more buildings than it otherwise would.
- The original year field is kept unchanged in `buildings_clean.csv`.
