"""11 — Phase 4 step 1+2: honest validation harness + per-horizon ensemble.

Addresses the v2.x finding that the HPO CV number (0.0361) was optimistic
because HPO tuned and reported on the SAME folds (decision days 830-869).

This script uses a CLEAN, HPO-UNCONTAMINATED validation set: rolling-origin
folds whose test windows are at decision days 770-829 — all strictly BEFORE
the HPO folds (830-869). The tuned params are FIXED here (no re-tuning), so
evaluating them on these folds is an honest generalisation estimate.

Then it builds a per-horizon ensemble (LightGBM + sNaive + ES) with weights
chosen on the clean validation, for ALL 10 horizons.

The 60-day holdout (870-929) stays SEALED — opened only by script 13.

Outputs:
  outputs/metrics/phase4-honest-eval.md / .json
  outputs/forecasts/phase4-val-pred.csv
"""

from __future__ import annotations
import json, sys, time, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_wide_daily
from src.models import lightgbm_train, lightgbm_predict, snaive_forecast, es_forecast, LGB_PARAMS_BASELINE

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUTM = ROOT / "work" / "outputs" / "metrics"
OUTF = ROOT / "work" / "outputs" / "forecasts"
MODELS = ROOT / "work" / "outputs" / "models"
for p in (OUTM, OUTF):
    p.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
REPORTING_LAG = 3
# Clean validation fold cutoffs (decision-day start of each 10-day test window),
# all < 830 so they are HPO-uncontaminated.
VAL_FOLD_STARTS = [770, 780, 790, 800, 810, 820]   # 6 folds × 10 days = 60 val decision days
TEST_M = 10


def bucket(errs_by_h):
    out = {}
    for lab, lo, hi in [("overall", 1, 10), ("1-5d", 1, 5), ("6-10d", 6, 10)]:
        vals = [e for h in range(lo, hi + 1) for e in errs_by_h.get(h, [])]
        out[lab] = float(np.mean(np.square(vals))) if vals else float("nan")
    return out


def main():
    t0 = time.time()
    wide = load_wide_daily()
    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    dates = wide["midday_day"].to_list()
    date_to_idx = {d: i for i, d in enumerate(dates)}

    params = json.loads((MODELS / "lightgbm-tuned-best-params.json").read_text())
    print(f"Tuned params: objective={params.get('objective')}, p={params.get('tweedie_variance_power')}, "
          f"n_est={params.get('n_estimators')}, leaves={params.get('num_leaves')}")
    print(f"Clean validation folds (HPO-uncontaminated): decision-day starts {VAL_FOLD_STARTS} "
          f"= {dates[VAL_FOLD_STARTS[0]]} .. {dates[VAL_FOLD_STARTS[-1]+TEST_M-1]}\n")

    # Collect predictions on validation folds
    rows = []
    lgb_err = {h: [] for h in HORIZONS}
    sn_err = {h: [] for h in HORIZONS}
    es_err = {h: [] for h in HORIZONS}

    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        fm_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
        th = time.time()
        for fs in VAL_FOLD_STARTS:
            test_days = list(range(fs, fs + TEST_M))
            train_mask = fm_idx < (fs - h)            # labels strictly before fold start
            test_mask = np.isin(fm_idx, test_days)
            Xtr = fm.filter(pl.Series(train_mask)).select(feat)
            ytr = fm.filter(pl.Series(train_mask))["y"].to_numpy()
            Xte = fm.filter(pl.Series(test_mask)).select(feat)
            yte = fm.filter(pl.Series(test_mask))["y"].to_numpy()
            dte = fm.filter(pl.Series(test_mask))["decision_day"].to_list()
            if len(Xtr) < 30 or len(Xte) == 0:
                continue
            model, cmap = lightgbm_train(Xtr, ytr, params=params)
            p_lgb = lightgbm_predict(model, Xte, mapping=cmap)
            for D, yt, pl_ in zip(dte, yte, p_lgb):
                Di = date_to_idx[D]
                obs = y_full[: Di - (REPORTING_LAG - 1)]
                p_sn = float(snaive_forecast(obs, horizon=10, start_offset=REPORTING_LAG + 1)[h - 1])
                try:
                    p_es = float(es_forecast(obs, horizon=10, start_offset=REPORTING_LAG + 1)[h - 1])
                except Exception:
                    p_es = float(np.nanmean(obs))
                lgb_err[h].append(yt - pl_); sn_err[h].append(yt - p_sn); es_err[h].append(yt - p_es)
                rows.append({"h": h, "decision_day": str(D), "y_true": float(yt),
                             "lgb": float(pl_), "snaive": p_sn, "es": p_es})
        print(f"  h={h:2d}: {len(VAL_FOLD_STARTS)} folds  ({time.time()-th:.0f}s)")

    pred = pl.DataFrame(rows)
    pred.write_csv(OUTF / "phase4-val-pred.csv")

    lgb_b = bucket(lgb_err); sn_b = bucket(sn_err); es_b = bucket(es_err)

    # ---- Per-horizon ensemble: grid search 3-way weights on validation ----
    grid = np.linspace(0, 1, 11)
    ens_err = {h: [] for h in HORIZONS}
    weights = {}
    for h in HORIZONS:
        sub = pred.filter(pl.col("h") == h)
        yt = sub["y_true"].to_numpy(); L = sub["lgb"].to_numpy(); S = sub["snaive"].to_numpy(); E = sub["es"].to_numpy()
        best = (1.0, 0.0, 0.0); best_mse = float(np.mean((yt - L) ** 2))
        for wl in grid:
            for ws in grid:
                we = round(1 - wl - ws, 4)
                if we < -1e-9:
                    continue
                blend = wl * L + ws * S + we * E
                m = float(np.mean((yt - blend) ** 2))
                if m < best_mse:
                    best_mse, best = m, (float(wl), float(ws), float(we))
        weights[h] = {"w_lgb": best[0], "w_snaive": best[1], "w_es": best[2], "val_mse": best_mse}
        blend = best[0] * L + best[1] * S + best[2] * E
        ens_err[h] = list(yt - blend)
    ens_b = bucket(ens_err)

    dur = time.time() - t0

    # HPO-contaminated CV (from hpo json) for the honesty comparison
    hpo = json.loads((OUTM / "hpo-results.json").read_text())["best_summary"]
    # Holdout (sealed test) from the saved holdout run
    ho = json.loads((OUTM / "holdout-final.json").read_text())

    summary = {
        "validation_folds_decision_days": f"{dates[VAL_FOLD_STARTS[0]]} .. {dates[VAL_FOLD_STARTS[-1]+TEST_M-1]}",
        "n_val_folds": len(VAL_FOLD_STARTS),
        "lightgbm_tuned": {"clean_val": lgb_b,
                           "hpo_contaminated_cv": {"overall": hpo["mse_overall_mean"], "1-5d": hpo["mse_1_5d_mean"], "6-10d": hpo["mse_6_10d_mean"]},
                           "holdout": ho["lightgbm"]},
        "snaive": {"clean_val": sn_b, "holdout": {"overall": ho["snaive"]["mse_overall"], "1-5d": ho["snaive"]["mse_1_5d"], "6-10d": ho["snaive"]["mse_6_10d"]}},
        "es": {"clean_val": es_b, "holdout": {"overall": ho["es"]["mse_overall"], "1-5d": ho["es"]["mse_1_5d"], "6-10d": ho["es"]["mse_6_10d"]}},
        "ensemble": {"clean_val": ens_b, "per_horizon_weights": weights},
        "duration_seconds": dur,
    }
    (OUTM / "phase4-honest-eval.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    md = ["# Phase 4 — honest validation + ensemble (step 1+2)", "",
          f"Generated: {datetime.now().isoformat(timespec='seconds')} · {dur:.0f}s", "",
          f"**Clean validation** = 6 rolling folds, decision days {summary['validation_folds_decision_days']} "
          "(all BEFORE the HPO folds 830-869, so HPO-uncontaminated). Tuned params fixed (no re-tuning).",
          "Holdout (60 days) still sealed — opened by script 13.", "",
          "## A. HPO over-optimism, quantified (LightGBM, tuned)", "",
          "| | overall | 1-5d | 6-10d |", "|---|---:|---:|---:|",
          f"| HPO-reported CV (contaminated) | {hpo['mse_overall_mean']:.4f} | {hpo['mse_1_5d_mean']:.4f} | {hpo['mse_6_10d_mean']:.4f} |",
          f"| Clean validation (this run) | {lgb_b['overall']:.4f} | {lgb_b['1-5d']:.4f} | {lgb_b['6-10d']:.4f} |",
          f"| Holdout (sealed test) | {ho['lightgbm']['mse_overall']:.4f} | {ho['lightgbm']['mse_1_5d']:.4f} | {ho['lightgbm']['mse_6_10d']:.4f} |",
          "",
          "If clean-validation ≈ holdout but both >> HPO-reported CV, that confirms the HPO number was optimistic (it tuned on the folds it reported).",
          "",
          "## B. All methods on the clean validation", "",
          "| Method | overall | 1-5d | 6-10d |", "|---|---:|---:|---:|",
          f"| LightGBM (tuned) | {lgb_b['overall']:.4f} | {lgb_b['1-5d']:.4f} | {lgb_b['6-10d']:.4f} |",
          f"| sNaive | {sn_b['overall']:.4f} | {sn_b['1-5d']:.4f} | {sn_b['6-10d']:.4f} |",
          f"| ES | {es_b['overall']:.4f} | {es_b['1-5d']:.4f} | {es_b['6-10d']:.4f} |",
          f"| **Ensemble (per-h weighted)** | **{ens_b['overall']:.4f}** | **{ens_b['1-5d']:.4f}** | **{ens_b['6-10d']:.4f}** |",
          "",
          "## C. Per-horizon ensemble weights (chosen on clean validation)", "",
          "| h | w_LGB | w_sNaive | w_ES | val MSE |", "|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        w = weights[h]
        md.append(f"| {h} | {w['w_lgb']:.1f} | {w['w_snaive']:.1f} | {w['w_es']:.1f} | {w['val_mse']:.4f} |")
    md += ["",
           f"Ensemble vs LightGBM-alone on clean validation: "
           f"overall {(lgb_b['overall']-ens_b['overall'])/lgb_b['overall']*100:+.1f}%, "
           f"1-5d {(lgb_b['1-5d']-ens_b['1-5d'])/lgb_b['1-5d']*100:+.1f}%, "
           f"6-10d {(lgb_b['6-10d']-ens_b['6-10d'])/lgb_b['6-10d']*100:+.1f}%.",
           "", "Next: script 12 (Fourier features) then script 13 (final sealed-holdout confirmation + attribution table)."]
    (OUTM / "phase4-honest-eval.md").write_text("\n".join(md))

    print(f"\n=== Clean-validation results ({dur:.0f}s) ===")
    print(f"  LightGBM tuned : overall {lgb_b['overall']:.4f} | 1-5d {lgb_b['1-5d']:.4f} | 6-10d {lgb_b['6-10d']:.4f}")
    print(f"  (HPO said CV    : overall {hpo['mse_overall_mean']:.4f} | 1-5d {hpo['mse_1_5d_mean']:.4f} | 6-10d {hpo['mse_6_10d_mean']:.4f})")
    print(f"  (holdout said   : overall {ho['lightgbm']['mse_overall']:.4f} | 1-5d {ho['lightgbm']['mse_1_5d']:.4f} | 6-10d {ho['lightgbm']['mse_6_10d']:.4f})")
    print(f"  sNaive         : overall {sn_b['overall']:.4f} | 1-5d {sn_b['1-5d']:.4f} | 6-10d {sn_b['6-10d']:.4f}")
    print(f"  Ensemble       : overall {ens_b['overall']:.4f} | 1-5d {ens_b['1-5d']:.4f} | 6-10d {ens_b['6-10d']:.4f}")
    print(f"\nReport: {OUTM / 'phase4-honest-eval.md'}")


if __name__ == "__main__":
    main()
