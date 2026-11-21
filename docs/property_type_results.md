# Updated property-type results

The selected general model is MAE gradient boosting. Held-out MAE is 19.71 for that general model and 19.71 kBtu/ft²/year for the portfolio. The paired portfolio-minus-general difference is 0.00, with a 95% building-bootstrap interval of [-0.01, 0.02]. The interval includes zero, so this test does not establish an overall improvement. The property-type median baseline has MAE 19.79; the selected general model lowers that by 0.4%. Its paired general-minus-baseline 95% interval is [-0.60, 0.49]. Validation-selected routes: Self-Storage Facility: MAE gradient boosting. All remaining types use the selected general model, including types with insufficient specialist training or validation support.

## Recommendations by type

| property_type | train_buildings | validation_buildings | test_buildings | general_MAE | specialist_MAE | portfolio_MAE | selected_route |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multifamily Housing | 1854 | 698 | 772 | 10.46 | 10.86 | 10.46 | general |
| Office | 391 | 157 | 134 | 16.72 | 17.71 | 16.72 | general |
| Non-Refrigerated Warehouse | 195 | 80 | 81 | 11.17 | 10.65 | 11.17 | general |
| Manufacturing/Industrial Plant | 161 | 91 | 60 | 90.89 | 88.04 | 90.89 | general |
| Distribution Center | 85 | 42 | 30 | 11.55 | 10.95 | 11.55 | general |
| Retail Store | 76 | 33 | 30 | 23.30 | 23.13 | 23.30 | general |
| Supermarket/Grocery Store | 44 | 20 | 25 | 63.76 | Not fitted | 63.76 | general |
| Other | 55 | 26 | 23 | 34.53 | Not fitted | 34.53 | general |
| Hotel | 44 | 20 | 21 | 25.61 | Not fitted | 25.61 | general |
| Self-Storage Facility | 77 | 24 | 21 | 5.81 | 6.11 | 6.11 | absolute_boosting |
| Parking | 43 | 18 | 20 | 12.76 | Not fitted | 12.76 | general |

## Design and filters

All years of a building remain together. One fixed split allocates approximately 50% of buildings to training, 20% to validation, 10% to interval calibration, and 20% to testing. The same building assignments are used in all three cohorts. The broad cohort requires valid positive area and EUI, joined weather, and EUI at or below the existing 2,000 review cap. The common-types cohort retains types with at least 100 distinct training buildings. The primary stable-single-building cohort requires exactly one reported building in each retained record, floor area at least 1,000 ft², and no existing cross-year characteristic review flags. These are sensitivity cohorts, not a search for the lowest test error. Different cohort errors concern different populations.

## Model portfolio

Compare a training-only property-type median baseline, ridge regression, random forest, and absolute-error gradient boosting. All use the same allowed inputs (the median baseline uses type alone). Choose the general family by validation MAE. Fit specialists only with at least 60 training and 20 validation buildings of that type; choose their family on validation MAE. Route a type to its specialist only when validation MAE improves at least 5% over the selected general model on the same validation rows. Otherwise fall back to the general model. These thresholds are practical prespecified settings, not guarantees of adequate power. Models remain frozen after training. Test results do not alter the routes. Per-type recommendations are provisional because several candidates are compared on one validation split; repeated nested grouped validation would assess selection stability.

## Property-type performance and fairness

The main chart compares the selected general model with the best validation-selected specialist on exactly the same held-out records within each type. It displays types with at least 20 test buildings; the CSV retains every evaluated type and flags limited test support. Whiskers resample entire buildings, preserving their annual records. Positive signed error means underprediction, which can raise a type's apparent audit priority. Normalized MAE divides by mean observed EUI within the type. Macro MAE gives each evaluated type equal weight, including small unstable groups. A positive-residual rate is a flagging diagnostic, not a false-positive rate: verified audit outcomes are unavailable. Differences by property type demonstrate performance disparities, not demographic fairness or discrimination. Absolute-error boosting and median baselines target typical (median) consumption rather than mean consumption; a right-skewed consumption distribution can therefore produce positive mean signed errors even for useful models.

## Feature importance

Permutation importance measures the increase in held-out MAE when one input block is shuffled. Location and weather columns are shuffled jointly within each block, and the portfolio reroutes when property type is shuffled. The CSV also includes selected specialists with at least 20 test buildings. Whiskers show variation over eight shuffles, not confidence intervals. Importance is model-specific and associative. Correlation between year, weather, and location can spread or mask importance; these plots do not establish causal drivers or identify a building's faulty equipment. Target-derived inputs (energy, emissions, ENERGY STAR scores, weather-normalized EUI) are excluded. Floor area enters as a logarithm.

## Prediction intervals

A separate calibration sample sets a 90% interval using each calibration building's maximum absolute annual error. The portfolio's EUI radius is 48.40; the observed held-out fraction of buildings with all recorded years inside their intervals is 92.7%. Coverage is marginal across exchangeable buildings, not guaranteed within every property type. Intervals may be wide. They are prediction intervals, not confidence intervals for avoidable waste.

## Audit screening and manager communication

The manager example is a held-out building: none of its years trained or selected its prediction model. The demonstration ranking requires all three years, positive residuals in at least two years, and positive mean annual consumption gap. Its score is the mean of (actual EUI − predicted EUI) × that year's area; negative annual gaps are retained. Ratios above five require data review. The example is the highest-scoring supported-type candidate without that ratio flag. Show the manager the property type, annual actual and expected use, uncertainty, percent difference, and total consumption gap. A recurring positive gap motivates reviewing meters, reporting boundaries, occupancy, schedules, and equipment. It neither proves waste nor establishes program eligibility. The new ranking covers held-out primary-cohort buildings only; the existing full-sample OOF ranking remains a legacy output. A full-population portfolio ranking requires nested building-grouped predictions.

## Scope and next steps

This experiment estimates generalization to unseen buildings in the recorded 2023–2025 period, not future-year forecasting. Cross-year quality filters use reported characteristics across the available period; future deployment must compute eligibility using only data available at scoring time. Interpretation of program year and incomplete latest-year coverage follows the existing project's caveats. The Boston paper has not been supplied, so these figures are not claimed to reproduce its analysis. Review its target, sample, validation design, and figure definitions before claiming a replication.

## Reproduce

```bash
python scripts/03_clean_buildings.py
python scripts/05_build_model_table.py
python scripts/11_property_type_results.py
pytest -q tests
```

See `outputs/property_type_results/results_report.html` for the self-contained illustrated report. All new tables and figures are in that folder.