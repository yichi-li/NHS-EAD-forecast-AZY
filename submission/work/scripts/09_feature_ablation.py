"""09 — Feature ablation by family (Phase 3.3).

Identify which feature families carry real signal vs are noise/redundant.
For each family, drop it and re-train LightGBM with the best HPO params.
Compare MSE_overall to the "all features" baseline.

Families:
  - y_lag        (15 cols)  — autoregressive on y, shifted >= reporting lag
  - y_rolling    (8 cols)   — rolling mean/std of y, shifted >= reporting lag
  - x_lag        (1044 cols)— all x at lags 0/1/7
  - x_rolling    (300 cols) — rolling mean/std of top-50 x metrics
  - causal_aggregate (6 cols)— upstream/concurrent/downstream pressure mean/std
  - calendar     (10 cols)  — dow / month / holidays / etc.

Outputs:
  outputs/metrics/feature-ablation.md
  outputs/metrics/feature-ablation.json
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
    HoldoutSplit, RollingOriginCV, FoldResult, mse, summarise_per_horizon, train_row_mask_for_fold,
)
from src.data import load_wide_daily  # noqa: E402
from src.models import LGB_PARAMS_BASELINE, lightgbm_train, lightgbm_predict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "work" / "data"
OUT_METRICS = PROJECT_ROOT / "work" / "outputs" / "metrics"
OUT_MODELS = PROJECT_ROOT / "work" / "outputs" / "models"
OUT_METRICS.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))


def family_of(col: str) -> str:
    if col.startswith("y_lag_"):
        return "y_lag"
    if col.startswith("y_rmean_") or col.startswith("y_rstd_"):
        return "y_rolling"
    if any(col.endswith(f"_lag_{k}") for k in (0, 1, 7)):
        return "x_lag"
    if "_rmean_" in col or "_rstd_" in col:
        return "x_rolling"
    if col in {
        "upstream_pressure_mean", "upstream_pressure_std",
        "concurrent_pressure_mean", "concurrent_pressure_std",
        "downstream_pressure_mean", "downstream_pressure_std",
    }:
        return "causal_aggregate"
    if col in {
        "dow", "dom", "wom", "month", "year", "is_weekend",
        "is_uk_bank_holiday", "day_of_year",
        "is_christmas_period", "is_flu_season",
    }:
        return "calendar"
    return "other"


def evaluate(params, folds, date_to_idx, drop_family: str | None = None) -> dict:
    results: list[FoldResult] = []
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        decision_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat_cols = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
        if drop_family is not None:
            feat_cols = [c for c in feat_cols if family_of(c) != drop_family]

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


def main():
    t0 = time.time()
    wide = load_wide_daily()
    cv_slice, _ = HoldoutSplit().split(wide.shape[0])
    folds = RollingOriginCV().folds(cv_slice.stop)
    date_to_idx = {d: i for i, d in enumerate(wide["midday_day"].to_list())}

    # Load best HPO params, fall back to baseline if HPO not yet run
    best_params_path = OUT_MODELS / "lightgbm-tuned-best-params.json"
    if best_params_path.exists():
        params = json.loads(best_params_path.read_text())
        print(f"Using HPO-tuned params from {best_params_path.name}")
    else:
        params = deepcopy(LGB_PARAMS_BASELINE)
        params["tweedie_variance_power"] = 1.5
        print(f"HPO params not found — using Tweedie p=1.5 default")

    families = ["y_lag", "y_rolling", "x_lag", "x_rolling", "causal_aggregate", "calendar"]

    # 1. All features (the reference)
    print("\nEvaluating ALL features (reference)...")
    all_summary = evaluate(params, folds, date_to_idx, drop_family=None)
    ref_mse = all_summary["mse_overall_mean"]
    print(f"  MSE_overall = {ref_mse:.4f}")

    # 2. Drop each family
    results = {"_all_features": all_summary}
    for fam in families:
        print(f"\nDropping {fam!r} ...")
        ts = time.time()
        summary = evaluate(params, folds, date_to_idx, drop_family=fam)
        results[fam] = summary
        delta = summary["mse_overall_mean"] - ref_mse
        delta_pct = delta / ref_mse * 100
        print(f"  MSE_overall = {summary['mse_overall_mean']:.4f}  Δ = {delta:+.4f}  ({delta_pct:+.1f}% vs all-features)  ({time.time()-ts:.0f}s)")

    duration = time.time() - t0

    # Rank: most-essential families have biggest positive delta when dropped
    ranking = sorted(
        [(fam, results[fam]["mse_overall_mean"]) for fam in families],
        key=lambda kv: kv[1], reverse=True,
    )

    # Save
    (OUT_METRICS / "feature-ablation.json").write_text(json.dumps(results, indent=2, default=str))

    # Markdown
    md = [
        "# Feature ablation by family — Phase 3.3",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')} · Duration: {duration:.0f}s",
        "",
        "Method: drop each feature family from the input set, retrain LightGBM with the "
        "same hyperparameters (HPO-tuned if available, else Tweedie p=1.5 default), "
        "measure MSE_overall delta vs all-features baseline.",
        "",
        "## All-features reference",
        "",
        f"- **MSE_overall** = {ref_mse:.4f}",
        f"- **MSE_1-5d**    = {all_summary['mse_1_5d_mean']:.4f}",
        f"- **MSE_6-10d**   = {all_summary['mse_6_10d_mean']:.4f}",
        "",
        "## Drop-one-family results",
        "",
        "| Family dropped | MSE_overall | Δ vs all | Δ % | Interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for fam in families:
        s = results[fam]
        delta = s["mse_overall_mean"] - ref_mse
        delta_pct = delta / ref_mse * 100
        if delta_pct > 5:
            interp = "**essential** — dropping hurts a lot"
        elif delta_pct > 1:
            interp = "useful — small contribution"
        elif delta_pct > -1:
            interp = "neutral — could be removed without loss"
        else:
            interp = "**redundant** — dropping IMPROVES (noise!)"
        md.append(
            f"| {fam} | {s['mse_overall_mean']:.4f} | {delta:+.4f} | {delta_pct:+.1f}% | {interp} |"
        )
    md.append("")
    md.append("## Ranking by essentialness (largest MSE on drop = most essential)")
    md.append("")
    for i, (fam, m) in enumerate(ranking, 1):
        delta_pct = (m - ref_mse) / ref_mse * 100
        md.append(f"{i}. **{fam}** (MSE without it: {m:.4f}, +{delta_pct:.1f}% vs all)")
    md.append("")
    md.append("## Take-aways")
    md.append("")
    # Identify essential, redundant
    essential = [fam for fam, m in ranking if (m - ref_mse) / ref_mse > 0.05]
    redundant = [fam for fam, m in ranking if (m - ref_mse) / ref_mse < -0.01]
    if essential:
        md.append(f"- **Essential families (must keep):** {', '.join(essential)}")
    if redundant:
        md.append(f"- **Redundant / noisy (consider dropping):** {', '.join(redundant)}")
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append("- Single-family drop assumes families are independent — true interactions between families could be missed.")
    md.append("- Higher-order ablation (drop pairs of families) is left to v3.")
    md.append("- Result depends on hyperparameters; if HPO changes substantially, re-run.")
    md.append("")
    (OUT_METRICS / "feature-ablation.md").write_text("\n".join(md))
    print(f"\nReport: {OUT_METRICS / 'feature-ablation.md'}")


if __name__ == "__main__":
    main()
