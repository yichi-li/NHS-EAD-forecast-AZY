"""10 — Final holdout evaluation (Phase 3.3.x).

This is the ONE-SHOT evaluation on the 60-day holdout that was reserved at
the start of v2. Untouched by all prior development; opened ONCE here.

Procedure:
- Load HPO-tuned best params.
- For each h ∈ 1..10:
    - Train ONE model on ALL CV-pool data (decision_day < cv_end - h)
    - Predict on holdout rows (decision_day ∈ [cv_end - h, total_days - h))
    - These rows have label_day ∈ [cv_end, total_days), i.e., the holdout window.
- Aggregate MSE_overall, MSE_1-5d, MSE_6-10d.

Outputs:
  outputs/metrics/holdout-final.md
  outputs/metrics/holdout-final.json
  outputs/forecasts/holdout-final-pred.csv
  outputs/models/lightgbm-final-h{h}.txt
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import HoldoutSplit, mse  # noqa: E402
from src.data import load_wide_daily  # noqa: E402
from src.models import LGB_PARAMS_BASELINE, snaive_forecast, es_forecast, lightgbm_train, lightgbm_predict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "work" / "data"
OUT_METRICS = PROJECT_ROOT / "work" / "outputs" / "metrics"
OUT_MODELS = PROJECT_ROOT / "work" / "outputs" / "models"
OUT_FORECASTS = PROJECT_ROOT / "work" / "outputs" / "forecasts"
for p in [OUT_METRICS, OUT_MODELS, OUT_FORECASTS]:
    p.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
REPORTING_LAG = 3  # contest rule: at D, observable y up to y(D - REPORTING_LAG)


def main():
    t0 = time.time()
    wide = load_wide_daily()
    total_days = wide.shape[0]
    cv_slice, holdout_slice = HoldoutSplit().split(total_days)
    cv_end = cv_slice.stop
    holdout_start = holdout_slice.start

    print(f"Total dev days: {total_days}")
    print(f"CV pool used for training: [0, {cv_end})  ({cv_end} days)")
    print(f"Holdout (final eval): [{holdout_start}, {total_days})  ({total_days - holdout_start} days)")
    print()

    # Load HPO params
    best_params_path = OUT_MODELS / "lightgbm-tuned-best-params.json"
    if best_params_path.exists():
        params = json.loads(best_params_path.read_text())
        print(f"Using HPO-tuned params from {best_params_path.name}")
    else:
        params = deepcopy(LGB_PARAMS_BASELINE)
        params["tweedie_variance_power"] = 1.5
        print(f"HPO params not found — falling back to Tweedie p=1.5")
    print()

    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    wide_dates = wide["midday_day"].to_list()
    date_to_idx = {d: i for i, d in enumerate(wide_dates)}

    # Holdout test decision days: predict at each D in [cv_end, total_days)
    # such that y(D+1..D+10) is still inside dev. We use D ∈ [cv_end - 0, total_days - 1].
    # Constraint: D + h <= total_days - 1 → D <= total_days - h - 1
    # For h=10, D <= total_days - 11 → D <= 919 if total = 930
    # Holdout starts at index 870. Take D ∈ [870, 919] = 50 decision days
    holdout_decision_days = list(range(holdout_start, total_days - max(HORIZONS)))
    print(f"Holdout decision days: {len(holdout_decision_days)} days "
          f"({wide_dates[holdout_decision_days[0]]} .. {wide_dates[holdout_decision_days[-1]]})")
    print()

    # ----------------------------------------------------------
    # Naive baselines on holdout
    # ----------------------------------------------------------
    print("=== Naive baselines on holdout ===")
    naive_rows = []
    sn_sq = []
    es_sq = []
    sn_15, sn_610 = [], []
    es_15, es_610 = [], []
    for D in holdout_decision_days:
        # At decision day D, observable y up to y(D - REPORTING_LAG). Predict D+1..D+10.
        observable = y_full[: D - (REPORTING_LAG - 1)]
        p_sn = snaive_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)
        try:
            p_es = es_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)
        except Exception:
            p_es = np.full(10, np.nanmean(observable))
        for h_idx, h in enumerate(HORIZONS):
            y_t = float(y_full[D + h])
            y_sn = float(p_sn[h_idx])
            y_es = float(p_es[h_idx])
            sn_sq.append((y_t - y_sn) ** 2)
            es_sq.append((y_t - y_es) ** 2)
            if h <= 5:
                sn_15.append((y_t - y_sn) ** 2)
                es_15.append((y_t - y_es) ** 2)
            else:
                sn_610.append((y_t - y_sn) ** 2)
                es_610.append((y_t - y_es) ** 2)
            naive_rows.append({
                "decision_day": str(wide_dates[D]), "h": h, "label_day": str(wide_dates[D + h]),
                "y_true": y_t, "snaive": y_sn, "es": y_es,
            })
    sn_mse_all = float(np.mean(sn_sq))
    es_mse_all = float(np.mean(es_sq))
    sn_mse_15 = float(np.mean(sn_15))
    sn_mse_610 = float(np.mean(sn_610))
    es_mse_15 = float(np.mean(es_15))
    es_mse_610 = float(np.mean(es_610))
    print(f"  sNaive MSE_overall = {sn_mse_all:.4f}  (1-5d={sn_mse_15:.4f}, 6-10d={sn_mse_610:.4f})")
    print(f"  ES     MSE_overall = {es_mse_all:.4f}  (1-5d={es_mse_15:.4f}, 6-10d={es_mse_610:.4f})")

    # ----------------------------------------------------------
    # LightGBM final per-h on holdout
    # ----------------------------------------------------------
    print("\n=== LightGBM (HPO-tuned) per-h on holdout ===")
    lgb_rows = []
    lgb_sq_all = []
    lgb_sq_15 = []
    lgb_sq_610 = []
    per_h_mse = {}
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        fm_decision_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat_cols = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]

        # Train: all rows with decision_day + h <= cv_end (so label_day < cv_end + 1)
        # Equivalently: decision_day < cv_end - h + 1 = cv_end - h + 1
        train_mask = fm_decision_idx < cv_end - h + 1
        # Holdout rows: decision_day in holdout_decision_days
        test_mask = np.isin(fm_decision_idx, holdout_decision_days)

        X_train = fm.filter(pl.Series(train_mask)).select(feat_cols)
        y_train = fm.filter(pl.Series(train_mask))["y"].to_numpy()
        X_test = fm.filter(pl.Series(test_mask)).select(feat_cols)
        y_test = fm.filter(pl.Series(test_mask))["y"].to_numpy()
        test_dec_days = fm.filter(pl.Series(test_mask))["decision_day"].to_list()

        model, col_map = lightgbm_train(X_train, y_train, params=params)
        y_pred = lightgbm_predict(model, X_test, mapping=col_map)
        model.save_model(str(OUT_MODELS / f"lightgbm-final-h{h}.txt"))

        sq = (y_test - y_pred) ** 2
        per_h_mse[h] = float(np.mean(sq))
        lgb_sq_all.extend(sq.tolist())
        if h <= 5:
            lgb_sq_15.extend(sq.tolist())
        else:
            lgb_sq_610.extend(sq.tolist())

        for D, y_t, y_p in zip(test_dec_days, y_test, y_pred):
            lgb_rows.append({
                "decision_day": str(D), "h": h, "label_day": str(wide_dates[date_to_idx[D] + h]),
                "y_true": float(y_t), "y_pred": float(y_p),
            })
        print(f"  h={h:2d}: trained on {len(X_train)} rows, MSE on {len(X_test)} holdout rows = {per_h_mse[h]:.4f}")

    lgb_mse_all = float(np.mean(lgb_sq_all))
    lgb_mse_15 = float(np.mean(lgb_sq_15))
    lgb_mse_610 = float(np.mean(lgb_sq_610))

    duration = time.time() - t0
    print(f"\n=== FINAL HOLDOUT NUMBERS ({duration:.0f}s) ===")
    print(f"  sNaive:   MSE_overall = {sn_mse_all:.4f}  | 1-5d = {sn_mse_15:.4f}  | 6-10d = {sn_mse_610:.4f}")
    print(f"  ES:       MSE_overall = {es_mse_all:.4f}  | 1-5d = {es_mse_15:.4f}  | 6-10d = {es_mse_610:.4f}")
    print(f"  LightGBM: MSE_overall = {lgb_mse_all:.4f}  | 1-5d = {lgb_mse_15:.4f}  | 6-10d = {lgb_mse_610:.4f}")
    print(f"  LightGBM vs sNaive: {(sn_mse_all-lgb_mse_all)/sn_mse_all*100:+.1f}%")
    print(f"  LightGBM vs ES:     {(es_mse_all-lgb_mse_all)/es_mse_all*100:+.1f}%")

    # Save outputs
    out = {
        "holdout_days": total_days - holdout_start,
        "n_holdout_decision_days": len(holdout_decision_days),
        "first_holdout_decision_day": str(wide_dates[holdout_decision_days[0]]),
        "last_holdout_decision_day": str(wide_dates[holdout_decision_days[-1]]),
        "snaive": {"mse_overall": sn_mse_all, "mse_1_5d": sn_mse_15, "mse_6_10d": sn_mse_610},
        "es": {"mse_overall": es_mse_all, "mse_1_5d": es_mse_15, "mse_6_10d": es_mse_610},
        "lightgbm": {
            "mse_overall": lgb_mse_all, "mse_1_5d": lgb_mse_15, "mse_6_10d": lgb_mse_610,
            "per_h": per_h_mse,
            "params": params,
        },
        "duration_seconds": duration,
    }
    (OUT_METRICS / "holdout-final.json").write_text(json.dumps(out, indent=2, default=str))
    pl.DataFrame(naive_rows).write_csv(OUT_FORECASTS / "holdout-final-naive-pred.csv")
    pl.DataFrame(lgb_rows).write_csv(OUT_FORECASTS / "holdout-final-lgb-pred.csv")

    md = [
        "# Final holdout evaluation — Phase 3.3.x",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')} · Duration: {duration:.0f}s",
        "",
        f"**THE 60-DAY HOLDOUT IS NOW OPENED** (was untouched from v2 start until this script).",
        "",
        f"- Holdout period: **{out['first_holdout_decision_day']} .. {wide_dates[total_days-1]}**",
        f"- Decision days evaluated: {len(holdout_decision_days)}",
        f"- (D, h) pairs: {len(holdout_decision_days) * len(HORIZONS)}",
        "",
        "## Final numbers",
        "",
        "| Model | MSE_overall | MSE_1-5d | MSE_6-10d |",
        "|---|---:|---:|---:|",
        f"| **sNaive** | {sn_mse_all:.4f} | {sn_mse_15:.4f} | {sn_mse_610:.4f} |",
        f"| **ES (AutoETS s=7)** | {es_mse_all:.4f} | {es_mse_15:.4f} | {es_mse_610:.4f} |",
        f"| **LightGBM (HPO-tuned)** | **{lgb_mse_all:.4f}** | **{lgb_mse_15:.4f}** | **{lgb_mse_610:.4f}** |",
        "",
        f"**LightGBM vs sNaive:** {(sn_mse_all-lgb_mse_all)/sn_mse_all*100:+.1f}%",
        f"**LightGBM vs ES:**     {(es_mse_all-lgb_mse_all)/es_mse_all*100:+.1f}%",
        "",
        "## Per-horizon MSE (LightGBM)",
        "",
        "| h | MSE | Bucket |",
        "|---:|---:|---|",
    ]
    for h in HORIZONS:
        bucket = "1-5d" if h <= 5 else "6-10d"
        md.append(f"| {h} | {per_h_mse[h]:.4f} | {bucket} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append(f"- This MSE is the most honest estimate of v2 model performance on data the model never saw.")
    md.append(f"- Comparison with CV MSE (from `baseline-v2-comparison.md` / `hpo-results.md`):")
    md.append(f"  - If holdout >> CV: we overfit the CV folds during HPO. Use simpler params or different CV scheme.")
    md.append(f"  - If holdout ≈ CV: CV is a reliable estimator. Trust the HPO conclusions.")
    md.append(f"  - If holdout < CV: holdout period is easier than CV period (lucky). Don't be complacent.")
    md.append("")
    (OUT_METRICS / "holdout-final.md").write_text("\n".join(md))
    print(f"\nReport: {OUT_METRICS / 'holdout-final.md'}")


if __name__ == "__main__":
    main()
