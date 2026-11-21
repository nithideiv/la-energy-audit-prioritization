"""Build the canonical report from verified step-11/12 outputs.

Run scripts 03, 05, 06, 11, 12, then this script. No model is fitted here.
The final report never combines legacy OOF figures with the fixed-test experiment.
"""
from pathlib import Path
import base64
import html
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import load_config
from property_models import LABELS
from audit_reporting import file_sha256, verify_analysis_files

SOURCE = ROOT / "outputs/property_type_results"
OUT = ROOT / "outputs/final_report"
PRIMARY = "stable_single_building"


def main():
    manifest = verify_analysis_files(SOURCE)
    if file_sha256(ROOT / "data/processed/model_table.csv") != manifest["input_sha256"]:
        raise ValueError("The model table changed after fitting. Rerun scripts 06, 11, 12, and 13.")
    eda = json.loads((ROOT / "outputs/eda_manifest.json").read_text())
    if eda["input_sha256"] != manifest["input_sha256"] or eda["cdd_figure_sha256"] != file_sha256(ROOT / "outputs/figures/cdd_vs_site_eui.png"):
        raise ValueError("Exploratory scatterplot belongs to another run. Rerun scripts 06 and 13.")
    extras = json.loads((SOURCE / "extras_manifest.json").read_text())
    if extras["analysis_manifest_sha256"] != file_sha256(SOURCE / "run_manifest.json"):
        raise ValueError("Step 12 belongs to another analysis run. Rerun scripts 12 and 13.")
    if extras["cohort"] != PRIMARY:
        raise ValueError("Final report uses the stable-single-building cohort. Rerun script 12 without --cohort.")
    for name, checksum in extras["artifact_sha256"].items():
        if file_sha256(SOURCE / name) != checksum:
            raise ValueError(f"Step-12 artifact changed: {name}. Rerun scripts 12 and 13.")
    cfg = load_config()
    settings = cfg["property_type_analysis"]
    summary = json.loads((SOURCE / "analysis_summary.json").read_text())
    primary = next(c for c in summary["cohorts"] if c["cohort"] == PRIMARY)
    all_metrics = pd.read_csv(SOURCE / "test_metrics.csv")
    metrics = all_metrics[all_metrics.cohort.eq(PRIMARY)]
    types = pd.read_csv(SOURCE / "per_type_test_metrics.csv")
    types = types[types.cohort.eq(PRIMARY)].sort_values("test_buildings", ascending=False)
    rec = pd.read_csv(SOURCE / "recommendations_by_type.csv")
    predictions = pd.read_csv(SOURCE / "heldout_predictions.csv", dtype={"building_id": str})
    predictions = predictions[predictions.cohort.eq(PRIMARY)]
    assert predictions.split.eq("test").all()
    assert predictions.general_model.eq(primary["selected_general"]).all()
    portfolio = metrics[metrics.strategy.eq("selected_portfolio")].iloc[0]
    general = metrics[metrics.strategy.eq("selected_general")].iloc[0]
    baseline = metrics[metrics.strategy.eq("general_property_type_median")].iloc[0]
    assert np.isclose(abs(predictions.site_eui - predictions.pred_portfolio).mean(), portfolio.MAE)
    supported = types[~types.limited_test_support]
    limited = int(types.limited_test_support.sum())
    selected = types[types.selected_route.ne("general")]
    chosen_types = "; ".join(f"{r.property_type}: {LABELS[r.selected_route]}" for r in selected.itertuples())
    if not chosen_types:
        chosen_types = "No specialist met the validation improvement threshold"
    diff = primary["portfolio_minus_general_test_MAE"]
    lo, hi = primary["paired_difference_ci95"]
    verdict = ("The interval includes zero, so the test does not establish an overall improvement from specialization."
               if lo <= 0 <= hi else "The interval excludes zero; interpret this conditional on the selected models and this split.")
    baseline_lo, baseline_hi = primary["general_minus_baseline_ci95"]
    baseline_verdict = ("The general model is not distinguishable from the type-median baseline in this test."
                        if baseline_lo <= 0 <= baseline_hi else "The paired general-minus-baseline interval excludes zero.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "building_cards").mkdir(exist_ok=True)
    figure_map = [
        (SOURCE / "figures/02_filter_experiments.png", "01_filter_experiments.png"),
        (SOURCE / "figures/07_candidate_models.png", "02_general_candidates.png"),
        (SOURCE / "figures/01_general_vs_specialist.png", "03_general_vs_specialist.png"),
        (SOURCE / "figures/08_recommendation_by_type.png", "04_recommendations.png"),
        (SOURCE / "figures/11_actual_vs_predicted_log.png", "05_actual_vs_predicted.png"),
        (SOURCE / "figures/12_importance_by_type.png", "06_feature_importance.png"),
        (SOURCE / "figures/09_selection_fairness.png", "07_property_type_fairness.png"),
        (SOURCE / "figures/10_interval_coverage_by_type.png", "08_interval_coverage.png"),
        (ROOT / "outputs/figures/cdd_vs_site_eui.png", "09_cdd_vs_site_eui.png"),
    ]
    for source, name in figure_map:
        shutil.copyfile(source, OUT / "figures" / name)
    for bid in extras["card_ids"]:
        name = f"{bid}_audit_card.png"
        shutil.copyfile(SOURCE / "figures/building_cards" / name, OUT / "building_cards" / name)
    for name in ["test_metrics.csv", "per_type_test_metrics.csv", "validation_model_selection.csv",
                 "recommendations_by_type.csv", "feature_block_importance.csv", "interval_coverage_by_type.csv",
                 "selection_fairness_by_type.csv", "building_card_metadata.csv", "heldout_screen.csv"]:
        shutil.copyfile(SOURCE / name, OUT / "tables" / name)
    retention = []
    for cohort in summary["cohorts"]:
        counts = cohort["split_counts"]
        retention.append({"Cohort": cohort["cohort"],
                          "Retained rows": sum(c["rows"] for c in counts.values()),
                          "Retained buildings": sum(c["buildings"] for c in counts.values()),
                          **{f"{name.title()} buildings": counts[name]["buildings"]
                             for name in ["train", "validation", "calibration", "test"]}})
    retention = pd.DataFrame(retention)
    retention.to_csv(OUT / "tables/cohort_retention.csv", index=False)
    recommendations = types[["property_type", "train_buildings", "validation_buildings", "test_buildings",
                             "general_MAE", "specialist_MAE", "portfolio_MAE", "selected_route",
                             "selection_reason", "limited_test_support"]].copy()
    recommendations["recommended_model"] = recommendations.selected_route.map(
        lambda route: LABELS[primary["selected_general"]] if route == "general" else LABELS[route])
    recommendations.to_csv(OUT / "tables/recommendations_with_support.csv", index=False)
    bullets = [
        f"Primary evaluation: {len(predictions):,} annual records from {predictions.building_id.nunique():,} unseen buildings; stable single-building cohort.",
        f"General model selected on validation MAE: {LABELS[primary['selected_general']]}.",
        f"Test MAE: median baseline {baseline.MAE:.3f}; general {general.MAE:.3f}; portfolio {portfolio.MAE:.3f} kBtu/ft²/year.",
        f"Portfolio minus general MAE: {diff:+.4f}; 95% paired building-bootstrap interval [{lo:+.4f}, {hi:+.4f}]. {verdict}",
        f"General minus median-baseline MAE: {primary['general_minus_baseline_test_MAE']:+.4f}; interval [{baseline_lo:+.4f}, {baseline_hi:+.4f}]. {baseline_verdict}",
        f"Validation-selected specialist routes: {chosen_types}. All other routes fall back to the selected general model.",
    ]
    def paragraph(s):
        return f"<p>{html.escape(s)}</p>"
    def table(df):
        return df.to_html(index=False, float_format=lambda v: f"{v:,.3f}", na_rep="Not fitted", border=0)
    def picture(relative, caption):
        b64 = base64.b64encode((OUT / relative).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    sections = ["<h1>LA building energy audit prioritization</h1>",
                '<p class="subtitle">Final results · validation-selected models evaluated on unseen buildings</p>',
                '<div class="scope">All primary model comparisons below use the same stable-single-building test records. The filter-sensitivity and exploratory weather figures explicitly use their own stated cohorts. Legacy OOF and log-target model figures are not included.</div>',
                "<h2>1. Main findings</h2>", *(paragraph(s) for s in bullets),
                "<h2>2. Three filters and retained sample sizes</h2>",
                paragraph("Filtering changes which records enter an experiment; model grouping changes which model fits those records. This report compares three actual filtering cohorts, using fixed building assignments across them. No filter is selected because it produced a favorable test score."),
                paragraph(f"All eligible: positive area and EUI, joined weather, EUI at or below the existing {cfg['site_eui_model_max']:,} review cap. Common types: retain types with at least {settings['common_type_min_train_buildings']} distinct training buildings. Stable single-building (primary): exactly one reported building, area at least 1,000 ft², and no cross-year characteristic review flags. The filters do not claim to identify efficiently operated buildings."),
                table(retention),
                picture("figures/01_filter_experiments.png", "Figure 1. Actual sample-filter comparisons. Compare methods within each cohort; cross-cohort MAE differences also reflect different building populations. Macro MAE weights types equally, including small unstable groups."),
                "<h2>3. General candidates and selection protocol</h2>",
                paragraph("All years of a building stay together. Approximately 50% of buildings train models, 20% select candidates, 10% calibrate intervals, and 20% evaluate final performance. Preprocessing is fitted on training records. Models stay frozen after training; evaluation records never choose the general family, specialist family, or routing rule. This is one fixed held-out experiment, not an OOF or nested-CV estimate."),
                paragraph(f"A specialist requires at least {settings['min_specialist_train_buildings']} training and {settings['min_specialist_validation_buildings']} validation buildings. Each type's best specialist is selected on validation MAE, then used only if it improves over the selected general model by at least {100*settings['min_validation_gain_fraction']:.0f}% on those validation records. Types lacking support or sufficient gain use the general model. These are practical thresholds, not power guarantees. Repeated nested grouped validation is a future stability check, not a claim of this analysis."),
                picture("figures/02_general_candidates.png", "Figure 2. General candidate models on identical primary test records. The highlighted family was chosen using validation MAE, not the displayed test bars. Original-pipeline 'lowest MAE' statements apply only to the original candidate set."),
                "<h2>4. General versus per-type models and recommendations</h2>",
                paragraph(f"The comparison shows the best validation-selected specialist even when the frozen portfolio falls back to the general model. Display requires at least {settings['min_report_test_buildings']} test buildings; {len(supported)} types meet that threshold, and {limited} have limited individual evaluation support. Sample size never changes a previously chosen route. Detailed CSVs retain all types and fallback reasons."),
                picture("figures/03_general_vs_specialist.png", "Figure 3. General versus specialist MAE by exact property type on the same test records. Whiskers resample whole buildings. No specialist bar means insufficient training/validation support, rather than zero error."),
                picture("figures/04_recommendations.png", "Figure 4. Validation-selected routing recommendations. A small test error difference is not proof that a route is superior. Test uncertainty is descriptive and does not change routing."),
                table(recommendations[~recommendations.limited_test_support][["property_type", "train_buildings", "validation_buildings", "test_buildings", "general_MAE", "specialist_MAE", "portfolio_MAE", "recommended_model"]]),
                "<h2>5. Predicted versus reported consumption</h2>",
                paragraph("The diagonal indicates perfect predictions. MAE, RMSE, and R² are computed in original EUI units on all test records. Absolute-error boosting predicts typical (median) consumption, so mean signed error can be positive in a skewed distribution. Weak within-type prediction is an explicit limitation, even when MAE is competitive."),
                picture("figures/05_actual_vs_predicted.png", "Figure 5. General and portfolio predictions, residuals, and three example property types. Axes are linear and limited to each panel's combined 99th percentile for visibility; residual histogram displays −120 to +120. Metrics include the complete test sample. Similarity to a specific Boston paper has not been verified because that paper was not supplied."),
                "<h2>6. Feature importance</h2>",
                paragraph(f"The general-model plots refer to {LABELS[primary['selected_general']]} trained for the primary cohort. Importance is measured on unseen primary-cohort buildings. Single-column permutations and joint weather-block permutations answer different questions; correlated inputs can share importance. Whiskers show variation over repeated shuffles, not confidence intervals. These are predictive associations, not causal equipment diagnoses."),
                picture("figures/06_feature_importance.png", "Figure 6. Importance of individual inputs and input blocks within each adequately evaluated property type, using the same selected general model. Negative values indicate no stable gain under this permutation diagnostic."),
                "<h2>7. Performance and screening disparities across property types</h2>",
                paragraph("Compare normalized MAE, signed error, and screening rates across building uses. Positive signed error means underprediction and may increase a type's flagging rate. Rate differences can also reflect real consumption differences, area-based prioritization, and missing operational variables. They do not establish discrimination or false-positive disparities without audit outcomes. Error and coverage tables include denominators and uncertainty."),
                picture("figures/07_property_type_fairness.png", "Figure 7. Property-type performance and audit-screening rates for the frozen portfolio and selected general model. This is a building-use performance analysis, not a demographic fairness claim."),
                "<h2>8. Prediction intervals and manager-facing explanations</h2>",
                paragraph("Intervals use separate calibration buildings. Each calibration score is the building's maximum annual absolute error, targeting simultaneous coverage of the available years for a new exchangeable building. The constant radius is an exact finite-sample order statistic; the scaled sensitivity analysis uses the same procedure after dividing by expected EUI (floor 5 kBtu/ft²/year). The target is 90% marginal building-level coverage, not a guarantee within every type."),
                picture("figures/08_interval_coverage.png", "Figure 8. Observed constant and scaled interval coverage by property type, with 95% Wilson intervals. Coverage method and error-bar definitions are distinct from MAE reference bands."),
                paragraph("Manager cards use: " + extras["interval_definition"] + ". This is a prediction interval, not ±MAE or a confidence interval for savings. Each card uses exact named property-type counts; no pooled rare-type group is relabeled as a specific use."),
                paragraph("A building meets the screening rule only with all three years, positive residuals in at least two years, and a positive average annual consumption gap. The score averages each year's (actual EUI − predicted EUI) × that year's floor area, retaining negative annual gaps. The screen does not require exceeding the interval. Ratios above five require data review first. Rankings here cover held-out primary-cohort buildings only; they are not the legacy full-population ranking or statutory audit eligibility."),
    ]
    for bid in extras["card_ids"]:
        sections.append(picture(f"building_cards/{bid}_audit_card.png", f"Building {bid}. Expected use comes from models fitted on other buildings. Review metering, reporting boundaries, schedules, occupancy, and process loads before treating the gap as an opportunity. Consumption gaps are not verified savings."))
    sections += ["<h2>9. Exploratory weather plot and data review</h2>",
                 picture("figures/09_cdd_vs_site_eui.png", "Figure 9. The broad eligible modeling sample on a logarithmic EUI axis, using the existing review cap. Extreme-EUI records remain in a separate review CSV. Many points share weather because there are only nine station-year weather combinations; this scatterplot is descriptive."),
                 "<h2>10. Reproduce and interpret</h2>",
                 paragraph("Run scripts 03, 05, 06, 11, 12, and 13 in order using the existing local downloads/geocodes. Older scripts 07–10 remain available for the legacy OOF and peer analyses, but their metrics and rankings must not be substituted into this report. The report checks hashes to detect mixed analysis artifacts. The Boston-paper comparison and verified audit outcomes remain future work."),
                 paragraph("Cross-year quality filtering uses reported characteristics across the available study period. The experiment addresses new buildings in that period, not future-year forecasting. Latest-year filing incompleteness, reporting errors, coarse weather matching, single-split model selection, and unmeasured operating schedules limit interpretation.")]
    style = "body{font:16px/1.65 system-ui,sans-serif;color:#213341;max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:34px}h2{color:#164c72;margin-top:40px}.subtitle,figcaption{color:#526575}.scope{padding:18px;background:#eef5fa;border-left:4px solid #2b6cb0}img{width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:13px}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:12px}th,td{padding:8px 10px;border-bottom:1px solid #dbe4ea;text-align:left}th{background:#eef3f8}p{max-width:1100px}"
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>LA energy audit: final results</title><style>' + style + '</style></head><body>' + "\n".join(sections) + '</body></html>'
    (OUT / "final_results_report.html").write_text(page)
    final_manifest = {"primary_cohort": PRIMARY, "general_model": primary["selected_general"],
                      "analysis_manifest_sha256": file_sha256(SOURCE / "run_manifest.json"),
                      "extras_manifest_sha256": file_sha256(SOURCE / "extras_manifest.json"),
                      "artifact_sha256": {str(p.relative_to(OUT)): file_sha256(p) for p in sorted(OUT.rglob("*"))
                                          if p.is_file() and p.name != "final_manifest.json"}}
    (OUT / "final_manifest.json").write_text(json.dumps(final_manifest, indent=2))
    (ROOT / "docs/final_results.md").write_text("# Final results\n\n" + "\n\n".join(bullets) +
        "\n\nOpen `outputs/final_report/final_results_report.html` for the coherent illustrated report. "
        "The tables, figures, card metadata, and checksums are in that directory.\n")
    print(f"Saved {OUT / 'final_results_report.html'}")
    print(f"Primary test sample: {predictions.building_id.nunique():,} buildings, {len(predictions):,} rows; general: {LABELS[primary['selected_general']]}")


if __name__ == "__main__":
    main()
