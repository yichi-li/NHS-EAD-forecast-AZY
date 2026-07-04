# Submission — NHS estimated avoidable deaths forecast

Self-contained, leakage-safe, reproducible forecasting pipeline for the SPHERE-PPL
NHS-EAD forecasting contest. Produces `pred_matrix.csv` and `mse_summary.csv` for
the 131 assessment periods (1 Oct 2025 – 17 Feb 2026, the organisers' amended
validation window).

## Contents

```
submission/
  README.md            this file
  REPORT.md            method + results (<= 1000 words)
  report.html          rendered report
  pred_matrix.csv      131 x 10 forecasts (the scored deliverable)
  mse_summary.csv      per-period MSE (1-5d, 6-10d)
  work/                the code, in the layout its paths expect
    main.py            deployment entry point (produces the two CSVs)
    run.sh             wrapper that sets DYLD_LIBRARY_PATH for LightGBM (macOS libomp)
    pyproject.toml     pinned dependencies (uv / pip)
    src/               data.py, features.py, models.py, feature_catalog.py
    scripts/           01_long_to_wide, 02_eda, 03_feature_catalog, 04_feature_engineering, ...
    tests/             leakage + correctness tests
    outputs/eda/05-lag-correlations.csv   dev-only lag-0 selection (see below)
  NHS-EAD-forecast-main/data/             put the two official CSVs here (see below)
```

## Data

The pipeline reads the two official long-format CSVs. They are not committed (large
/ Git-LFS). Place them here before running:

```
submission/NHS-EAD-forecast-main/data/turingAI_forecasting_challenge_dataset.csv
submission/NHS-EAD-forecast-main/data/turingAI_forecasting_challenge_validation_dataset.csv
```

(The path is resolved relative to `work/`, so running `work/main.py` from anywhere
finds the data at `submission/NHS-EAD-forecast-main/data/`.)

## Reproduce

```bash
cd submission/work
uv sync                                             # or: pip install -e . / from pyproject.toml
./run.sh python scripts/01_long_to_wide.py          # build the daily wide table (dev + validation)
./run.sh python gen_lag0.py                          # dev-only lag-0 selection -> outputs/eda/05-lag-correlations.csv
./run.sh python scripts/03_feature_catalog.py       # causal-role catalogue
./run.sh python scripts/04_feature_engineering.py   # per-horizon feature matrices
./run.sh python main.py                             # -> outputs/submission/{pred_matrix,mse_summary}.csv
```

The shipped `outputs/eda/05-lag-correlations.csv` lets you skip `gen_lag0.py`;
`04` refuses to run if that file is expected but missing (it will not silently
fall back to a different feature set).

## Leakage discipline (contest rules)

- A forecast for days D+1..D+10 uses only data up to midday day D.
- The target has a 3-day reporting lag: at day D the newest usable target is y(D-3);
  target lags therefore start at 3, and each training pair (d, y(d+h)) is used only
  when its label is already observed (d <= D-3-h).
- Feature selection (the lag-0 top-N metrics) and the causal-pressure z-score
  constants are fitted on the **development window only** (<= 30 Sep 2025), never on
  the assessment period.
- No external data; only the two provided CSVs are read.

Runtime for one 10-day forecast set is a few seconds, well within the 1-hour limit.
