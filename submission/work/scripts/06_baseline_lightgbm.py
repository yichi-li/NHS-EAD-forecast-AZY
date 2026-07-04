"""06 — Honest baseline (v2): direct per-horizon LightGBM, leakage-safe.

For each horizon h ∈ 1..10:
  1. Load `feature_matrix_h{h}.parquet` (one row per decision_day, label = y(D+h))
  2. For each of K rolling-origin folds (in the CV pool, holdout untouched):
       - train_mask = (decision_day_idx < fold.train_end_day - h)
       - test_mask  = decision_day_idx in fold.test_decision_days
       - Fit LightGBM(Tweedie) on train, predict on test
  3. Aggregate per (fold, h) MSE → mse_1_5d, mse_6_10d, mse_overall (mean ± std).

Also run sNaive and ES (AutoETS) baselines on the y series at the same decision
days, for direct comparison with the official ladder (DECISIONS.md D2).

Outputs:
  outputs/metrics/baseline-v2-comparison.md      — narrative + tables
  outputs/metrics/baseline-v2-metrics.json       — per-fold / per-horizon raw
  outputs/forecasts/baseline-v2-pred.csv         — every (fold, h, D) prediction
  outputs/models/lightgbm-baseline-v2-h{h}.txt   — saved boosters
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

import xgboost as xgb
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import (  # noqa: E402
    HOLDOUT_DAYS,
    HoldoutSplit,
    RollingOriginCV,
    Fold,
    FoldResult,
    mse,
    summarise_per_horizon,
    train_row_mask_for_fold,
)
from src.data import load_wide_daily  # noqa: E402
from src.models import (  # noqa: E402
    LGB_PARAMS_BASELINE,
    snaive_forecast,
    es_forecast,
    lightgbm_train,
    lightgbm_predict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "work" / "data"
OUT_MODELS = PROJECT_ROOT / "work" / "outputs" / "models"
OUT_FORECASTS = PROJECT_ROOT / "work" / "outputs" / "forecasts"
OUT_METRICS = PROJECT_ROOT / "work" / "outputs" / "metrics"
for p in [OUT_MODELS, OUT_FORECASTS, OUT_METRICS]:
    p.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
REPORTING_LAG = 3  # contest rule: at D, observable y up to y(D - REPORTING_LAG)


def main():
    t0 = time.time()

    # 1. Build holdout / CV split using the wide-daily date axis
    wide = load_wide_daily()
    total_days = wide.shape[0]
    ### holdout = last 60 days kept hidden to test model
    cv_slice, holdout_slice = HoldoutSplit().split(total_days)
    cv_end = cv_slice.stop                 # = 870 if total=930
    ### rolling origin cross validation
    ### i.e. many test train splits
    folds = RollingOriginCV().folds(cv_end)

    print(f"Total dev days     : {total_days}")
    print(f"CV pool indices    : [0, {cv_end})  ({cv_end} days)")
    print(f"Holdout indices    : [{cv_end}, {total_days})  ({total_days - cv_end} days)  ← UNTOUCHED here")
    print(f"\nCV folds (K={len(folds)}, each test_m=10 decision days):")
    for f in folds:
        test_dates = [str(wide['midday_day'][i]) for i in (f.test_decision_days[0], f.test_decision_days[-1])]
        print(f"  Fold {f.fold_idx}: train [0, {f.train_end_day})  test decision days {f.test_decision_days[0]}..{f.test_decision_days[-1]}  ({test_dates[0]} .. {test_dates[1]})")
    print()

    # The full y series, indexed by decision_day index in wide (= day index in dev)
    y_full = wide["estimated_avoidable_deaths"].to_numpy()

    # ------------------------------------------------------------------
    # 2. Naive baselines (sNaive, ES) on y series
    # ------------------------------------------------------------------
    print("=== sNaive + ES baselines ===")
    naive_results: list[FoldResult] = []
    es_results: list[FoldResult] = []
    naive_pred_rows = []

    for fold in folds:
        # Naive baselines forecast from the train cutoff
        train_y = y_full[: fold.train_end_day]

        for D_idx in fold.test_decision_days:
            # snaive / es forecast 10 steps ahead from the training cutoff
            # Adjusted: for each D in fold.test_decision_days, the "decision day"
            # is at index D, we predict y(D+1)..y(D+10)
            # Observable y at decision day D = y(0..D - REPORTING_LAG).
            # y_full[:D_idx - (REPORTING_LAG - 1)] gives indices [0..D_idx-REPORTING_LAG],
            # i.e. y(0..D-REPORTING_LAG), inclusive. The last observable is y(D-3).
            # Predicting D+1..D+10 means start_offset = REPORTING_LAG + 1 = 4 days after last observable.
            observable = y_full[: D_idx - (REPORTING_LAG - 1)]
            p_sn = snaive_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)
            try:
                p_es = es_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)
            except Exception:
                p_es = np.full(10, np.nanmean(observable))

            for h_idx, h in enumerate(HORIZONS):
                target_idx = D_idx + h
                if target_idx >= total_days:
                    continue
                y_true = float(y_full[target_idx])
                y_sn = float(p_sn[h_idx])
                y_es = float(p_es[h_idx])
                naive_results.append(FoldResult(fold.fold_idx, h, 1, (y_true - y_sn) ** 2, np.array([y_true]), np.array([y_sn])))
                es_results.append(FoldResult(fold.fold_idx, h, 1, (y_true - y_es) ** 2, np.array([y_true]), np.array([y_es])))
                naive_pred_rows.append({
                    "fold": fold.fold_idx,
                    "decision_day_idx": D_idx,
                    "decision_day": str(wide["midday_day"][D_idx]),
                    "h": h,
                    "label_day": str(wide["midday_day"][target_idx]),
                    "y_true": y_true,
                    "snaive": y_sn,
                    "es": y_es,
                })

    naive_summary = summarise_per_horizon(naive_results)
    es_summary = summarise_per_horizon(es_results)
    print(f"  sNaive  MSE_overall = {naive_summary['mse_overall_mean']:.4f} ± {naive_summary['mse_overall_std']:.4f}")
    print(f"  ES      MSE_overall = {es_summary['mse_overall_mean']:.4f} ± {es_summary['mse_overall_std']:.4f}")

    # ------------------------------------------------------------------
    # 3. LightGBM — direct forecasting per h
    # ------------------------------------------------------------------
    print(f"\n=== LightGBM (Tweedie, default-ish) — direct per-h, {len(HORIZONS)} models × {len(folds)} folds ===")
    lgb_results: list[FoldResult] = []
    lgb_pred_rows = []

    ### train 10 seperate lightGBM models, one per horizon, to avoid leakage
    for h in HORIZONS:
        fm_h = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        # Decision day index in `wide` is recoverable: each row's decision_day matches wide["midday_day"][i]
        # for some i. Build the mapping.
        wide_dates = wide["midday_day"].to_list()
        date_to_idx = {d: i for i, d in enumerate(wide_dates)}
        fm_decision_idx = np.array([date_to_idx[d] for d in fm_h["decision_day"].to_list()])

        feat_cols = [c for c in fm_h.columns if c not in ("decision_day", "label_day", "y")]

        t_h = time.time()
        for fold in folds:
            train_mask = train_row_mask_for_fold(fm_decision_idx, fold, h)
            # Test rows: only those whose decision_day is in fold.test_decision_days
            test_mask = np.isin(fm_decision_idx, fold.test_decision_days)

            X_train = fm_h.filter(pl.Series(train_mask)).select(feat_cols)
            y_train = fm_h.filter(pl.Series(train_mask))["y"].to_numpy()
            X_test = fm_h.filter(pl.Series(test_mask)).select(feat_cols)
            y_test = fm_h.filter(pl.Series(test_mask))["y"].to_numpy()
            test_dec_days = fm_h.filter(pl.Series(test_mask))["decision_day"].to_list()

            if len(X_train) < 30 or len(X_test) == 0:
                continue

            model, col_map = lightgbm_train(X_train, y_train)
            y_pred = lightgbm_predict(model, X_test, mapping=col_map)

            for D, y_t, y_p in zip(test_dec_days, y_test, y_pred):
                lgb_pred_rows.append({
                    "fold": fold.fold_idx,
                    "decision_day": str(D),
                    "h": h,
                    "label_day": str(wide["midday_day"][date_to_idx[D] + h]),
                    "y_true": float(y_t),
                    "y_pred": float(y_p),
                })

            mse_fh = mse(y_test, y_pred)
            lgb_results.append(FoldResult(fold.fold_idx, h, len(y_test), mse_fh, y_test, y_pred))

            # Save the model from the last fold for inspection
            if fold.fold_idx == len(folds) - 1:
                model.save_model(str(OUT_MODELS / f"lightgbm-baseline-v2-h{h}.txt"))
        print(f"  h={h:2d}: trained {len(folds)} folds  ({time.time()-t_h:.1f}s)")

    lgb_summary = summarise_per_horizon(lgb_results)

    ### XGBoost 
    ### https://arxiv.org/abs/1603.02754
    ### https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/param_tuning.html
    XGB_PARAMS = {
        ### regression using tweedie loss (robust to outliers and good for values near 0)
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.1,
        ### 2000 decision trees
        "n_estimators": 1500,
        "learning_rate": 0.05,
        ### how deep each tree can be
        "max_depth": 6,
        ### node needs at least 100 samples to be created
        "min_child_weight": 100,
        ### each tree sees 80% features
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        ### set seed for reproducibility
        "random_state": 42,
        "verbosity": 0,
    }

    ### collect results
    print(f"\n=== XGBoost (Tweedie) — direct per-h, {len(HORIZONS)} models × {len(folds)} folds ===")
    xgb_results: list[FoldResult] = []
    xgb_pred_rows = []

    ### loop through 1 - 10
    ### for each horizon load in feature matrix previously defined
    for h in HORIZONS:
        fm_h = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        wide_dates = wide["midday_day"].to_list()
        date_to_idx = {d: i for i, d in enumerate(wide_dates)}
        fm_decision_idx = np.array([date_to_idx[d] for d in fm_h["decision_day"].to_list()])
        feat_cols = [c for c in fm_h.columns if c not in ("decision_day", "label_day", "y")]

        t_h = time.time()
        for fold in folds:
            train_mask = train_row_mask_for_fold(fm_decision_idx, fold, h)
            test_mask = np.isin(fm_decision_idx, fold.test_decision_days)

            X_train = fm_h.filter(pl.Series(train_mask)).select(feat_cols).to_pandas()
            y_train = fm_h.filter(pl.Series(train_mask))["y"].to_numpy()
            X_test = fm_h.filter(pl.Series(test_mask)).select(feat_cols).to_pandas()
            y_test = fm_h.filter(pl.Series(test_mask))["y"].to_numpy()
            test_dec_days = fm_h.filter(pl.Series(test_mask))["decision_day"].to_list()

            if len(X_train) < 30 or len(X_test) == 0:
                continue

            ### create model
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)

            ### save predictions
            for D, y_t, y_p in zip(test_dec_days, y_test, y_pred):
                xgb_pred_rows.append({
                    "fold": fold.fold_idx,
                    "decision_day": str(D),
                    "h": h,
                    "label_day": str(wide["midday_day"][date_to_idx[D] + h]),
                    "y_true": float(y_t),
                    "y_pred": float(y_p),
                })

            ### score model
            mse_fh = mse(y_test, y_pred)
            xgb_results.append(FoldResult(fold.fold_idx, h, len(y_test), mse_fh, y_test, y_pred))

            if fold.fold_idx == len(folds) - 1:
                model.save_model(str(OUT_MODELS / f"xgboost-baseline-h{h}.json"))

        print(f"  h={h:2d}: trained {len(folds)} folds  ({time.time()-t_h:.1f}s)")

    xgb_summary = summarise_per_horizon(xgb_results)
    
    ### CatBoost
    ### https://catboost.ai/en/docs/concepts/python-reference_catboostregressor
    CAT_PARAMS = {
        ### Tweedie loss - same as XGBoost and LightGBM
        "loss_function": "Tweedie:variance_power=1.1",
        ### build 800 trees
        "iterations": 800,
        "learning_rate": 0.05,
        ### how deep each tree can be
        "depth": 6,
        ### node needs at least 100 samples to be created
        "min_data_in_leaf": 100,
        ### each tree sees 80% rows
        "subsample": 0.8,
        ### each tree sees 80% features
        "colsample_bylevel": 0.8,
        ### set seed for reproducibility
        "random_seed": 42,
        "verbose": 0,
    }

    ### store scores
    print(f"\n=== CatBoost (Tweedie) — direct per-h, {len(HORIZONS)} models × {len(folds)} folds ===")
    cat_results: list[FoldResult] = []
    cat_pred_rows = []

    ### loop through 1 - 10
    ### for each horizon load in feature matrix previously defined
    for h in HORIZONS:
        fm_h = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        wide_dates = wide["midday_day"].to_list()
        date_to_idx = {d: i for i, d in enumerate(wide_dates)}
        fm_decision_idx = np.array([date_to_idx[d] for d in fm_h["decision_day"].to_list()])
        ### get feature column names
        feat_cols = [c for c in fm_h.columns if c not in ("decision_day", "label_day", "y")]

        t_h = time.time()
        for fold in folds:
            train_mask = train_row_mask_for_fold(fm_decision_idx, fold, h)
            test_mask = np.isin(fm_decision_idx, fold.test_decision_days)

            X_train = fm_h.filter(pl.Series(train_mask)).select(feat_cols).to_pandas()
            y_train = fm_h.filter(pl.Series(train_mask))["y"].to_numpy()
            X_test = fm_h.filter(pl.Series(test_mask)).select(feat_cols).to_pandas()
            y_test = fm_h.filter(pl.Series(test_mask))["y"].to_numpy()
            test_dec_days = fm_h.filter(pl.Series(test_mask))["decision_day"].to_list()

            if len(X_train) < 30 or len(X_test) == 0:
                continue

            ### create and fit model
            model = CatBoostRegressor(**CAT_PARAMS)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            ### save predictions
            for D, y_t, y_p in zip(test_dec_days, y_test, y_pred):
                cat_pred_rows.append({
                    "fold": fold.fold_idx,
                    "decision_day": str(D),
                    "h": h,
                    "label_day": str(wide["midday_day"][date_to_idx[D] + h]),
                    "y_true": float(y_t),
                    "y_pred": float(y_p),
                })

            mse_fh = mse(y_test, y_pred)
            cat_results.append(FoldResult(fold.fold_idx, h, len(y_test), mse_fh, y_test, y_pred))

            if fold.fold_idx == len(folds) - 1:
                model.save_model(str(OUT_MODELS / f"catboost-baseline-h{h}.cbm"))

        print(f"  h={h:2d}: trained {len(folds)} folds  ({time.time()-t_h:.1f}s)")

    cat_summary = summarise_per_horizon(cat_results)

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    summary = {
        "n_folds": len(folds),
        "test_m_per_fold": len(folds[0].test_decision_days) if folds else 0,
        "n_horizons": len(HORIZONS),
        "holdout_days": HOLDOUT_DAYS,
        "snaive": naive_summary,
        "es": es_summary,
        "lightgbm": lgb_summary,
        "xgboost": xgb_summary,
        "catboost": cat_summary
    }
    (OUT_METRICS / "baseline-v2-metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    # Predictions CSV
    pl.DataFrame(naive_pred_rows).write_csv(OUT_FORECASTS / "baseline-v2-naive-pred.csv")
    pl.DataFrame(lgb_pred_rows).write_csv(OUT_FORECASTS / "baseline-v2-lgb-pred.csv")
    pl.DataFrame(xgb_pred_rows).write_csv(OUT_FORECASTS / "baseline-v2-xgb-pred.csv")
    pl.DataFrame(cat_pred_rows).write_csv(OUT_FORECASTS / "baseline-v2-catboost-pred.csv")
    duration = time.time() - t0

    # ------------------------------------------------------------------
    # 5. Markdown report
    # ------------------------------------------------------------------
    md = ["# Honest baseline — v2 (Phase 2.5.4)", ""]
    md.append(f"Generated: {datetime.now().isoformat(timespec='seconds')} · Duration: {duration:.1f}s")
    md.append("")
    md.append("**Setup (v2 — fixes the v1 leakage):**")
    md.append("- Per-horizon feature matrices (`feature_matrix_h{1..10}.parquet`), each leakage-safe (24/24 unit tests pass).")
    md.append("- Direct forecasting: one LightGBM model per h.")
    md.append("- Last 60 days of dev set carved as HOLDOUT (untouched here).")
    md.append(f"- {len(folds)} rolling-origin CV folds in the remaining {folds[-1].train_end_day + 10} days, each fold's test = 10 consecutive decision days × 10 horizons = 100 (D, h) pairs.")
    md.append("- LightGBM params: same Tweedie variance_power=1.1 block as v1, min_data_in_leaf=100, n_estimators=800.")
    md.append("")
    md.append("## MSE summary (mean ± std across folds)")
    md.append("")
    md.append("| Model | MSE_overall | MSE_1-5d | MSE_6-10d |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| **sNaive** | {naive_summary['mse_overall_mean']:.4f} ± {naive_summary['mse_overall_std']:.4f} | {naive_summary['mse_1_5d_mean']:.4f} ± {naive_summary['mse_1_5d_std']:.4f} | {naive_summary['mse_6_10d_mean']:.4f} ± {naive_summary['mse_6_10d_std']:.4f} |")
    md.append(f"| **ES (AutoETS s=7)** | {es_summary['mse_overall_mean']:.4f} ± {es_summary['mse_overall_std']:.4f} | {es_summary['mse_1_5d_mean']:.4f} ± {es_summary['mse_1_5d_std']:.4f} | {es_summary['mse_6_10d_mean']:.4f} ± {es_summary['mse_6_10d_std']:.4f} |")
    md.append(f"| **LightGBM (per-h)** | {lgb_summary['mse_overall_mean']:.4f} ± {lgb_summary['mse_overall_std']:.4f} | {lgb_summary['mse_1_5d_mean']:.4f} ± {lgb_summary['mse_1_5d_std']:.4f} | {lgb_summary['mse_6_10d_mean']:.4f} ± {lgb_summary['mse_6_10d_std']:.4f} |")
    md.append(f"| **XGBoost (per-h)** | {xgb_summary['mse_overall_mean']:.4f} ± {xgb_summary['mse_overall_std']:.4f} | {xgb_summary['mse_1_5d_mean']:.4f} ± {xgb_summary['mse_1_5d_std']:.4f} | {xgb_summary['mse_6_10d_mean']:.4f} ± {xgb_summary['mse_6_10d_std']:.4f} |")
    md.append(f"| **CatBoost (per-h)** | {cat_summary['mse_overall_mean']:.4f} ± {cat_summary['mse_overall_std']:.4f} | {cat_summary['mse_1_5d_mean']:.4f} ± {cat_summary['mse_1_5d_std']:.4f} | {cat_summary['mse_6_10d_mean']:.4f} ± {cat_summary['mse_6_10d_std']:.4f} |")
    md.append("## MSE per horizon (LightGBM)")
    md.append("")
    md.append("| h | MSE mean | MSE std | Notes |")
    md.append("|---:|---:|---:|---|")
    md.append("## MSE per horizon (XGBoost)")
    md.append("")
    md.append("| h | MSE mean | MSE std | Notes |")
    md.append("|---:|---:|---:|---|")
    for h in HORIZONS:
        per_h = xgb_summary["mse_per_h"].get(h, {})
        bucket = "1-5d" if h <= 5 else "6-10d"
        md.append(f"| {h} | {per_h.get('mean', float('nan')):.4f} | {per_h.get('std', float('nan')):.4f} | ({bucket}) |")
        for h in HORIZONS:
            per_h = lgb_summary["mse_per_h"].get(h, {})
            bucket = "1-5d" if h <= 5 else "6-10d"
            md.append(f"| {h} | {per_h.get('mean', float('nan')):.4f} | {per_h.get('std', float('nan')):.4f} | ({bucket}) |")
    md.append("")
    md.append("## Improvement of LightGBM over naive baselines")
    md.append("")
    sn = naive_summary["mse_overall_mean"]
    es = es_summary["mse_overall_mean"]
    lgb = lgb_summary["mse_overall_mean"]
    xgb_mse = xgb_summary["mse_overall_mean"] 
    md.append(f"- **vs sNaive:** {(sn - lgb) / sn * 100:+.1f}% lower MSE_overall")
    md.append(f"- **vs ES:**     {(es - lgb) / es * 100:+.1f}% lower MSE_overall")
    md.append("")
    md.append("## Comparison with v1 (leaky) baseline")
    md.append("")
    md.append("| | v1 (LEAKY) | v2 (honest) | Delta |")
    md.append("|---|---:|---:|---|")
    md.append(f"| sNaive MSE        | 0.0917 | {sn:.4f} | naive baseline shifts with fold composition |")
    md.append(f"| ES MSE            | 0.0762 | {es:.4f} | ditto |")
    md.append(f"| LightGBM MSE      | 0.0210 | {lgb:.4f} | the v1 number was inflated by horizon-mismatch leakage |")
    md.append(f"| LightGBM beats sNaive by | 77.1% | {(sn-lgb)/sn*100:.1f}% | honest improvement |")
    md.append("")
    md.append("## Caveats remaining (carried into Phase 3)")
    md.append("")
    md.append("1. **No HPO yet** — params are YJ_STU-style defaults. Phase 3.2 sweep next.")
    md.append("2. **No ensembling** — single per-h LightGBM. Phase 4.")
    md.append("3. **Loss function untested empirically** — Tweedie chosen because of Y skew=1.20; should A/B vs L2 / Huber in Phase 3.2.a.")
    md.append("4. **Holdout (last 60 days) still untouched** — opens only in Phase 3.3.x.")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `work/outputs/metrics/baseline-v2-metrics.json` — raw per-fold / per-horizon")
    md.append("- `work/outputs/forecasts/baseline-v2-naive-pred.csv` — sNaive + ES predictions")
    md.append("- `work/outputs/forecasts/baseline-v2-lgb-pred.csv` — LightGBM predictions")
    md.append(f"- `work/outputs/models/lightgbm-baseline-v2-h{{1..10}}.txt` — last-fold models")
    md.append("- `work/outputs/forecasts/baseline-v2-xgb-pred.csv` — XGBoost predictions")
    md.append(f"- `work/outputs/models/xgboost-baseline-h{{1..10}}.json` — last-fold models")

    out_md = OUT_METRICS / "baseline-v2-comparison.md"
    out_md.write_text("\n".join(md))

    print(f"=== Summary ({duration:.1f}s) ===")
    print(f"  sNaive:    MSE_overall = {sn:.4f}")
    print(f"  ES:        MSE_overall = {es:.4f}")
    print(f"  LightGBM:  MSE_overall = {lgb:.4f}")
    print(f"  XGBoost:   MSE_overall = {xgb_mse:.4f}")
    cat_mse = cat_summary["mse_overall_mean"]
    print(f"  CatBoost:  MSE_overall = {cat_mse:.4f}")
    print(f"  CatBoost vs sNaive:   {(sn-cat_mse)/sn*100:+.1f}%")
    print(f"  CatBoost vs LightGBM: {(lgb-cat_mse)/lgb*100:+.1f}%")
    print(f"  CatBoost vs XGBoost:  {(xgb_mse-cat_mse)/xgb_mse*100:+.1f}%")
    print(f"  LightGBM vs sNaive:  {(sn-lgb)/sn*100:+.1f}%")
    print(f"  XGBoost vs sNaive:   {(sn-xgb_mse)/sn*100:+.1f}%")
    print(f"  XGBoost vs LightGBM: {(lgb-xgb_mse)/lgb*100:+.1f}%")
    print(f"  Report: {out_md}")


if __name__ == "__main__":
    main()
