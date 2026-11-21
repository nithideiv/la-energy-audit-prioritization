# Final results

Primary evaluation: 3,680 annual records from 1,385 unseen buildings; stable single-building cohort.

General model selected on validation MAE: MAE gradient boosting.

Test MAE: median baseline 19.794; general 19.708; portfolio 19.713 kBtu/ft²/year.

Portfolio minus general MAE: +0.0045; 95% paired building-bootstrap interval [-0.0076, +0.0163]. The interval includes zero, so the test does not establish an overall improvement from specialization.

General minus median-baseline MAE: -0.0857; interval [-0.5963, +0.4854]. The general model is not distinguishable from the type-median baseline in this test.

Validation-selected specialist routes: Self-Storage Facility: MAE gradient boosting. All other routes fall back to the selected general model.

Open `outputs/final_report/final_results_report.html` for the coherent illustrated report. The tables, figures, card metadata, and checksums are in that directory.
