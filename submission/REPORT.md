# Forecasting NHS estimated avoidable deaths — method and results

## Task

We forecast the daily `estimated_avoidable_deaths` for each of the next ten days
from 220 NHS system-pressure metrics, and are scored by mean squared error over
the 1–5 day and 6–10 day horizons. The released assessment window is 1 Oct 2025
to 17 Feb 2026 (the organisers' amended validation dataset): 131 sliding 10-day
periods that span autumn and winter.

## Data and preprocessing

The raw long-format records are aggregated to a daily wide table using a midday
cut-off (entries up to 12:00 are assigned to the same day, later ones to the
next). The development window holds 930 days (16 Mar 2023 – 30 Sep 2025) with 348
metric-by-coverage columns; the released validation data adds three further
metrics from 1 Oct 2025, giving 351 metric columns plus the target on the
assessment build. This matches the reference preprocessing.

## Leakage discipline

Two contest rules drive the pipeline. First, a forecast for days D+1…D+10 may use
only data available up to midday on day D. Second, the target has a three-day
reporting lag: at day D the most recent usable target value is y(D−3). We build a
**separate feature matrix per horizon** so that no horizon can read a target value
that would not yet be observed at the forecast origin, and we restrict training
pairs (d, y(d+h)) to those whose label is already observed (d ≤ D−3−h). Thirty
unit tests guard these rules. We never use a metric value dated after the forecast
origin.

## Features

For each horizon we build: target lags y(D−3)…y(D−17) and rolling mean/standard
deviation of the target (windows 7/14/30/60, shifted by the reporting lag);
calendar features for the target day (day-of-week, month, UK bank holidays,
Christmas and flu-season flags — all known in
advance); per-metric lags and rolling statistics; and three causal-role pressure
aggregates (upstream / concurrent / downstream of ED boarding, standardised and
averaged). We compare a **full** feature set (1392 features) against a compact
**calendar+target-history** set (39 features that use no raw metric lags).

## Models and validation

We evaluate seasonal-naïve and exponential-smoothing baselines, LightGBM,
XGBoost and CatBoost (Tweedie loss), and ExtraTrees (Extremely Randomised Trees,
from scikit-learn), each on both feature sets.
Validation uses rolling-origin folds (always train on the past, score the future)
placed inside a **winter window carved from the development data** (Dec 2024 –
Feb 2025), because the held-out assessment period is winter and a summer holdout
is materially easier. We also re-check on a summer window. Reported figures are
out-of-sample; hyperparameters were tuned on a sub-set of folds and reported on
held-out folds, so the numbers are not inflated by tuning on the same data.

Two findings shaped the model. (1) Winter is much harder than summer — the best
out-of-sample MSE is roughly three times the summer figure, because winter
avoidable-death counts are about twice as high and more volatile. (2) The strong
configuration in winter is **ExtraTrees on the 39
calendar+target-history features**: adding the raw metric columns makes this model
worse on winter, whereas LightGBM benefits from the full feature set. The two
models therefore use complementary information and combine well. XGBoost and
CatBoost were evaluated on the same protocol but did not improve on this blend,
so they are not used in the final model. Results were stable across random seeds.

| Model (winter, out-of-sample) | MSE 1–5d | MSE 6–10d |
|---|--:|--:|
| Seasonal-naïve (floor) | 0.244 | 0.310 |
| LightGBM (full features) | 0.171 | 0.192 |
| ExtraTrees (calendar+history) | 0.153 | 0.170 |
| **Frozen framework (combined)** | **0.151** | **0.161** |

## Frozen submission

The submitted algorithm is a self-recalibrating framework. At each forecast origin
it re-fits both models on all data available up to that origin (recalibration is
permitted by the rules) and combines them per horizon. ExtraTrees
on the compact feature set is the backbone, blended with LightGBM on the full
feature set. The blend is weighted by horizon: a more even split at the short
horizons, where LightGBM's use of the system metrics helps, shifting to mostly
ExtraTrees at the longer horizons, where those metrics are stale
and calendar-plus-history carries the signal. On a per-origin winter dry-run that
reproduces the scoring exactly, this combination beats either model alone. The
horizon weights were fixed on this winter development dry-run, never on the
assessment data, and pure ExtraTrees at the long horizons is within about 0.001
MSE, so the result does not hinge on the exact split.
Producing one ten-day forecast takes a few seconds, well within the one-hour limit.
For a like-for-like comparison with entries that report a development backtest, we
also evaluate the framework on the most recent 173 development origins (the same
rolling-origin basis, all ten target days observed): MSE 1–5d = 0.040 and
6–10d = 0.046. On the released assessment window (131 periods, 1 Oct 2025 to
17 Feb 2026) it realises MSE 1–5d = 0.103 and 6–10d = 0.111. The three sets of
figures differ by season, not by model: the winter-only holdout above (0.151 /
0.161) is the deliberately hard stress test, the development backtest falls in the
lower-pressure spring and summer, and the released assessment window is winter-leaning.

## Reproducibility

The pipeline runs in order: build the daily table (`scripts/01`), the dev-only
lag-0 feature selection (`gen_lag0`), the causal-role catalogue (`03`), the
per-horizon feature matrices (`04`), then `main.py` to write `pred_matrix.csv` and
`mse_summary.csv`. `README.md` gives the exact folder layout and where the two
official CSVs go. All validation is leakage-tested. The forecasts included here are the 131 assessment
forecasts, produced by re-running `main.py` on the released validation data (the
development CSV plus the organisers' amended validation CSV, which ends 17 Feb 2026).

_Word count target ≤ 1000._
