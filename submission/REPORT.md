# Forecasting NHS estimated avoidable deaths — method and results

## Task

We forecast the daily `estimated_avoidable_deaths` for each of the next ten days
from 220 NHS system-pressure metrics, and are scored by mean squared error over
the 1–5 day and 6–10 day horizons. The assessment period (Oct 2025 – Mar 2026)
falls in winter.

## Data and preprocessing

The raw long-format records are aggregated to a daily wide table using a midday
cut-off (entries up to 12:00 are assigned to the same day, later ones to the
next), giving 930 days × 349 metric-by-coverage columns over 16 Mar 2023 –
30 Sep 2025. This matches the reference preprocessing.

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
Christmas and flu-season flags, and weekly/annual Fourier terms — all known in
advance); per-metric lags and rolling statistics; and three causal-role pressure
aggregates (upstream / concurrent / downstream of ED boarding, standardised and
averaged). We compare a **full** feature set (1383 features) against a compact
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
reproduces the scoring exactly, this combination beats either model alone.
Producing one ten-day forecast takes a few seconds, well within the one-hour limit.

## Reproducibility

The pipeline runs in order: build the daily table (`scripts/01`), classify metrics
by causal role (`03`), build the per-horizon feature matrices (`04`), then run
`main.py` to produce the official `pred_matrix.csv` and `mse_summary.csv`. All
validation is leakage-tested. The forecasts included here are a development-period
demonstration; the 173 assessment forecasts are produced by re-running `main.py`
on the released 6-June data, per the contest timeline.

_Word count target ≤ 1000._
