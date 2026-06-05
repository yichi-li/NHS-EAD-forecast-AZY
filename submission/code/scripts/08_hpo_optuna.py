"""08 — Hyperparameter tuning via Optuna (Phase 3.2.b).

Optimises LightGBM hyperparameters using the CV folds. The HOLDOUT
(last 60 days) is NEVER touched in this script.

Strategy:
- Loss = Tweedie (winner of Phase 3.2.a, power floating in [1.05, 1.95])
- Objective = minimise mean MSE across (fold, horizon) pairs
- Trials = 30 (Tweedie p=1.5 baseline = 0.0425; we want < 0.040 if possible)
- Sampler = TPE (Optuna default)

Outputs:
  outputs/metrics/hpo-results.md
  outputs/metrics/hpo-results.json
  outputs/models/lightgbm-tuned-best-params.json
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
import optuna
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import (  # noqa: E402
    HoldoutSplit,
    RollingOriginCV,
    FoldResult,
    mse,
    summarise_per_horizon,
    train_row_mask_for_fold,
)
from src.data import load_wide_daily  # noqa: E402
from src.models import LGB_PARAMS_BASELINE, lightgbm_train, lightgbm_predict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "work" / "data"
OUT_METRICS = PROJECT_ROOT / "work" / "outputs" / "metrics"
OUT_MODELS = PROJECT_ROOT / "work" / "outputs" / "models"
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
N_TRIALS = 30


# Caches (load once)
_FM_CACHE: dict[int, tuple] = {}


def get_fm_cached(h: int, date_to_idx):
    if h not in _FM_CACHE:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        decision_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat_cols = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
        _FM_CACHE[h] = (fm, decision_idx, feat_cols)
    return _FM_CACHE[h]


def evaluate_params(params: dict, folds, date_to_idx) -> dict:
    results: list[FoldResult] = []
    for h in HORIZONS:
        fm, decision_idx, feat_cols = get_fm_cached(h, date_to_idx)
        for fold in folds:
            train_mask = train_row_mask_for_fold(decision_idx, fold, h)
            test_mask = np.isin(decision_idx, fold.test_decision_days)
            X_train = fm.filter(pl.Series(train_mask)).select(feat_cols)
            y_train = fm.filter(pl.Series(train_mask))["y"].to_numpy()
            X_test = fm.filter(pl.Series(test_mask)).select(feat_cols)
            y_test = fm.filter(pl.Series(test_mask))["y"].to_numpy()
            if len(X_train) < 30 or len(X_test) == 0:
                continue
            model, col_map = lightgbm_train(X_train, y_train, params=params)
            y_pred = lightgbm_predict(model, X_test, mapping=col_map)
            results.append(FoldResult(fold.fold_idx, h, len(y_test), mse(y_test, y_pred), y_test, y_pred))
    return summarise_per_horizon(results)


def objective(trial, folds, date_to_idx):
    params = deepcopy(LGB_PARAMS_BASELINE)
    # Tweedie winner from Phase 3.2.a, with variance_power floating
    params["objective"] = "tweedie"
    params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.08, log=True)
    params["num_leaves"] = trial.suggest_int("num_leaves", 15, 511)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 20, 500)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.3, 1.0)
    params["subsample"] = trial.suggest_float("subsample", 0.3, 1.0)
    params["max_bin"] = trial.suggest_int("max_bin", 32, 255)
    params["n_estimators"] = trial.suggest_int("n_estimators", 200, 2000, step=100)

    summary = evaluate_params(params, folds, date_to_idx)
    # We optimise overall MSE
    return summary["mse_overall_mean"]


def main():
    t0 = time.time()
    wide = load_wide_daily()
    cv_slice, _ = HoldoutSplit().split(wide.shape[0])
    folds = RollingOriginCV().folds(cv_slice.stop)
    date_to_idx = {d: i for i, d in enumerate(wide["midday_day"].to_list())}

    print(f"HPO: {N_TRIALS} trials, TPE sampler, optimising mean MSE across "
          f"{len(folds)} folds × {len(HORIZONS)} horizons (= {len(folds) * len(HORIZONS)} model fits per trial)")
    print()

    # Quiet optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Custom callback to print progress
    def progress_callback(study, trial):
        bv = study.best_value
        print(f"  trial {trial.number:3d} done: MSE={trial.value:.4f}  best={bv:.4f}  ({trial.params})")

    study.optimize(
        lambda t: objective(t, folds, date_to_idx),
        n_trials=N_TRIALS,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    duration = time.time() - t0
    best_params_full = deepcopy(LGB_PARAMS_BASELINE)
    best_params_full["objective"] = "tweedie"
    best_params_full.update(study.best_params)

    print(f"\n=== Best ===")
    print(f"  MSE_overall = {study.best_value:.4f}")
    print(f"  vs baseline (tweedie p=1.5): 0.0425 → {study.best_value:.4f}  ({(0.0425-study.best_value)/0.0425*100:+.1f}%)")
    print(f"  Best params: {study.best_params}")
    print(f"  Duration: {duration:.0f}s")

    # Re-evaluate best params to get full breakdown
    best_summary = evaluate_params(best_params_full, folds, date_to_idx)

    # Save outputs
    out_json = {
        "n_trials": N_TRIALS,
        "best_value_mse_overall": study.best_value,
        "best_params": study.best_params,
        "best_summary": {k: v for k, v in best_summary.items() if k not in ("mse_per_h",)},
        "best_mse_per_h": best_summary["mse_per_h"],
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
        "duration_seconds": duration,
        "baseline_3_2_a_winner": {"config": "tweedie_p1.5", "mse_overall": 0.0425},
        "baseline_v2_default": {"config": "tweedie_p1.1", "mse_overall": 0.0450},
    }
    (OUT_METRICS / "hpo-results.json").write_text(json.dumps(out_json, indent=2, default=str))
    (OUT_MODELS / "lightgbm-tuned-best-params.json").write_text(json.dumps(best_params_full, indent=2))

    # Markdown
    md = [
        "# Hyperparameter tuning — Phase 3.2.b",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')} · Duration: {duration:.0f}s · Trials: {N_TRIALS}",
        "",
        "**Sampler:** Optuna TPE (seed=42)",
        "**Objective:** minimise mean MSE across CV folds × horizons (LightGBM with Tweedie loss).",
        "**Holdout:** untouched (Phase 3.3.x will open it).",
        "",
        "## Best trial",
        "",
        f"- **MSE_overall mean**: **{study.best_value:.4f}**  (± {best_summary['mse_overall_std']:.4f})",
        f"- **MSE_1-5d**: {best_summary['mse_1_5d_mean']:.4f} ± {best_summary['mse_1_5d_std']:.4f}",
        f"- **MSE_6-10d**: {best_summary['mse_6_10d_mean']:.4f} ± {best_summary['mse_6_10d_std']:.4f}",
        "",
        "### Best hyperparameters",
        "",
        "```",
    ]
    for k, v in sorted(study.best_params.items()):
        if isinstance(v, float):
            md.append(f"  {k:25s} = {v:.4f}")
        else:
            md.append(f"  {k:25s} = {v}")
    md.extend([
        "```",
        "",
        "## Progress over baselines",
        "",
        "| Stage | MSE_overall | Δ vs prior |",
        "|---|---:|---|",
        f"| v2 baseline (defaults, Tweedie p=1.1) | 0.0450 | — |",
        f"| Loss A/B winner (Tweedie p=1.5)       | 0.0425 | -5.6% |",
        f"| **HPO-tuned (this run)**              | **{study.best_value:.4f}** | **{(0.0425-study.best_value)/0.0425*100:+.1f}% vs p=1.5** |",
        "",
        "## All trials (sorted by MSE)",
        "",
        "| Trial | MSE | tweedie_p | lr | leaves | min_data | feat_frac | subsample | max_bin | n_est |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else 9e9)[:10]
    for t in sorted_trials:
        if t.value is None:
            continue
        p = t.params
        md.append(
            f"| {t.number} | {t.value:.4f} | {p.get('tweedie_variance_power', 0):.2f} | "
            f"{p.get('learning_rate', 0):.4f} | {p.get('num_leaves', 0)} | "
            f"{p.get('min_data_in_leaf', 0)} | {p.get('feature_fraction', 0):.2f} | "
            f"{p.get('subsample', 0):.2f} | {p.get('max_bin', 0)} | {p.get('n_estimators', 0)} |"
        )
    md.append("")
    md.append(f"(Top 10 of {N_TRIALS} trials.)")
    md.append("")
    (OUT_METRICS / "hpo-results.md").write_text("\n".join(md))
    print(f"\nReport: {OUT_METRICS / 'hpo-results.md'}")
    print(f"Best params: {OUT_MODELS / 'lightgbm-tuned-best-params.json'}")


if __name__ == "__main__":
    main()
