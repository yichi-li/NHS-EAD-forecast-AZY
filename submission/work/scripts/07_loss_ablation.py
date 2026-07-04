"""07 — Loss A/B (Phase 3.2.a): Tweedie vs L2 vs Huber for LightGBM.

Same per-horizon FE, same CV folds, same hyperparameters EXCEPT the objective.
Compares MSE_overall mean ± std across folds. Outputs:
  outputs/metrics/loss-ablation.md
  outputs/metrics/loss-ablation.json
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
OUT_METRICS.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))

LOSS_CONFIGS = {
    "tweedie_p1.1": {"objective": "tweedie", "tweedie_variance_power": 1.1},
    "tweedie_p1.5": {"objective": "tweedie", "tweedie_variance_power": 1.5},
    "regression_l2": {"objective": "regression", "metric": "rmse"},
    "huber": {"objective": "huber", "alpha": 0.9},
    "poisson": {"objective": "poisson"},
}


def run_one_config(name: str, loss_kw: dict, wide, folds, date_to_idx):
    params = deepcopy(LGB_PARAMS_BASELINE)
    params.update(loss_kw)
    # Remove keys that may clash with the new objective
    if loss_kw.get("objective") != "tweedie":
        params.pop("tweedie_variance_power", None)

    print(f"\n>> Config '{name}'  params: objective={params.get('objective')}", end="")
    if "tweedie_variance_power" in params:
        print(f" p={params['tweedie_variance_power']}", end="")
    if "alpha" in params:
        print(f" alpha={params['alpha']}", end="")
    print()

    t0 = time.time()
    results: list[FoldResult] = []
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        fm_decision_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat_cols = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]

        for fold in folds:
            train_mask = train_row_mask_for_fold(fm_decision_idx, fold, h)
            test_mask = np.isin(fm_decision_idx, fold.test_decision_days)
            X_train = fm.filter(pl.Series(train_mask)).select(feat_cols)
            y_train = fm.filter(pl.Series(train_mask))["y"].to_numpy()
            X_test = fm.filter(pl.Series(test_mask)).select(feat_cols)
            y_test = fm.filter(pl.Series(test_mask))["y"].to_numpy()

            if len(X_train) < 30 or len(X_test) == 0:
                continue

            model, col_map = lightgbm_train(X_train, y_train, params=params)
            y_pred = lightgbm_predict(model, X_test, mapping=col_map)
            results.append(FoldResult(fold.fold_idx, h, len(y_test), mse(y_test, y_pred), y_test, y_pred))

    summary = summarise_per_horizon(results)
    summary["config_name"] = name
    summary["params"] = {k: params.get(k) for k in ("objective", "tweedie_variance_power", "alpha", "metric") if k in params}
    summary["duration_seconds"] = time.time() - t0
    print(f"   MSE_overall = {summary['mse_overall_mean']:.4f} ± {summary['mse_overall_std']:.4f}  ({summary['duration_seconds']:.0f}s)")
    return summary


def main():
    wide = load_wide_daily()
    total_days = wide.shape[0]
    cv_slice, _ = HoldoutSplit().split(total_days)
    folds = RollingOriginCV().folds(cv_slice.stop)
    date_to_idx = {d: i for i, d in enumerate(wide["midday_day"].to_list())}

    all_summaries = {}
    for name, loss_kw in LOSS_CONFIGS.items():
        all_summaries[name] = run_one_config(name, loss_kw, wide, folds, date_to_idx)

    # Rank by MSE_overall_mean
    ranked = sorted(all_summaries.items(), key=lambda kv: kv[1]["mse_overall_mean"])

    print("\n=== Ranking by MSE_overall mean ===")
    for i, (name, s) in enumerate(ranked, 1):
        print(f"  #{i}  {name:<20} MSE = {s['mse_overall_mean']:.4f} ± {s['mse_overall_std']:.4f}    (MSE_1-5d={s['mse_1_5d_mean']:.4f}, MSE_6-10d={s['mse_6_10d_mean']:.4f})")

    # Save
    (OUT_METRICS / "loss-ablation.json").write_text(json.dumps(all_summaries, indent=2, default=str))

    # Markdown
    md = [
        "# Loss-function A/B — Phase 3.2.a",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Same per-horizon feature matrices, same CV folds, same hyperparameters EXCEPT the LightGBM `objective`. "
        "Goal: empirically validate the v1 default choice of Tweedie variance_power=1.1 vs alternatives.",
        "",
        "## Results (ranked by MSE_overall mean, lower = better)",
        "",
        "| Rank | Config | objective | param | MSE_overall mean | std | MSE_1-5d | MSE_6-10d |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, (name, s) in enumerate(ranked, 1):
        p = s["params"]
        obj = p.get("objective", "")
        extra = p.get("tweedie_variance_power", p.get("alpha", ""))
        md.append(
            f"| {i} | `{name}` | {obj} | {extra} | "
            f"{s['mse_overall_mean']:.4f} | {s['mse_overall_std']:.4f} | "
            f"{s['mse_1_5d_mean']:.4f} | {s['mse_6_10d_mean']:.4f} |"
        )
    md.append("")

    winner_name, winner = ranked[0]
    runner_name, runner = ranked[1]
    md.append(f"## Decision")
    md.append("")
    md.append(f"**Winner:** `{winner_name}` (MSE_overall = {winner['mse_overall_mean']:.4f})")
    md.append("")
    md.append(f"Runner-up: `{runner_name}` (MSE = {runner['mse_overall_mean']:.4f}, "
              f"delta = {(runner['mse_overall_mean']-winner['mse_overall_mean'])/winner['mse_overall_mean']*100:.1f}% worse).")
    md.append("")
    md.append("**Recommendation for Phase 3.2.b (HPO):** use the winning objective.")
    md.append("")
    md.append("## Validity of original v1 decision (D1)")
    md.append("")
    tweedie11 = all_summaries.get("tweedie_p1.1")
    md.append(f"- D1 (v1) chose Tweedie variance_power=1.1 based on Y skew=1.20 and YJ_STU's M5 recipe.")
    md.append(f"- Empirically: tweedie_p1.1 ranks #{[k for k, _ in ranked].index('tweedie_p1.1') + 1} out of {len(ranked)} configs.")
    if winner_name != "tweedie_p1.1":
        md.append(f"- The winning objective is `{winner_name}` instead. Suggests D1 should be revisited.")
    else:
        md.append(f"- Tweedie p=1.1 wins. D1 validated empirically.")
    md.append("")
    (OUT_METRICS / "loss-ablation.md").write_text("\n".join(md))
    print(f"\nReport: {OUT_METRICS / 'loss-ablation.md'}")


if __name__ == "__main__":
    main()
