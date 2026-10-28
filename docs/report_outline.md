# Final report outline

## 1. Introduction

- Why audit prioritization matters under limited budgets
- Research question and subgoals
- Interpretation boundary: screening, not causal retrofit estimates

## 2. Data

- EBEWE source, years, original/retained sample sizes
- NOAA stations, variables, units, flags, and coverage
- Census geocoding match rate and nearest-station distances
- Exact definition of reporting/compliance year: `Program Year` N = calendar-year-N consumption (per LADBS instructions; see README section 7), matched to same-year weather
- Note that PY 2025 filing is less complete than earlier years

## 3. Methods

- Cleaning and duplicate-resolution rules
- Predictors and leakage exclusions
- Grouped train/test logic and preprocessing within folds
- Baselines, linear regression, random forest, and metrics
- Out-of-fold residual and excess-use formulas
- Three-year eligibility rule and ranking stability
- Exploratory k-means: interpretation only, not ranking

## 4. Results

- EDA and missingness
- Building-only versus weather-added metrics
- Occupancy-added metrics (`*_occ` models) and the occupancy bootstrap in `report_stats.json`
- Residual checks by property type/year
- Ranked candidates and repeated positive residuals
- Top-10% Jaccard overlap across years/models

## 5. Limitations and ethics

- Occupancy is only a self-reported annual percent (tested in the `*_occ` models, with no significant gain); hourly schedules, vacancies, renovations, and special uses are still unobserved
- Reporting and geocoding error
- Three weather stations simplify microclimates
- Model residuals are not causal savings estimates
- Audit ranking should be reviewed by humans before outreach or spending
- PY 2025 (calendar-2025 data, filed in 2026) is less complete than earlier years (6,603 usable rows vs. 7,566 in 2023), so the three-year eligibility rule excludes more buildings than it otherwise would
- The highest-ranked buildings include special uses (data center, transportation terminal, refrigerated warehouse, parking). Their large residuals likely reflect operations the model can't see, or reporting errors, so rank 1 is not "least efficient"; candidates need human review before any outreach
- `check_data_first` in `audit_priority_ranking.csv` marks buildings whose actual Site EUI is more than 5x the predicted value; check these for reporting errors before treating them as candidates

## 6. Conclusion

- What weather did or did not add
- What patterns were stable
- Recommended next validation step: utility/occupancy review or on-site audit

