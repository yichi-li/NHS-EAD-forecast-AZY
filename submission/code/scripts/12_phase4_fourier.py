"""12 — Phase 4 step 3: Fourier seasonal features, evaluated on the SAME clean
validation folds as script 11 (HPO-uncontaminated, holdout sealed).

Builds Fourier-augmented per-horizon matrices (10 weekly+annual sin/cos terms
added to the calendar block), trains the same tuned LightGBM, and reports the
per-bucket MSE delta vs the non-Fourier LightGBM from script 11.

Outputs: outputs/metrics/phase4-fourier.md / .json
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
from src.feature_catalog import build_catalog
from src.features import build_features_for_horizon
from src.models import lightgbm_train, lightgbm_predict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUTM = ROOT / "work" / "outputs" / "metrics"
MODELS = ROOT / "work" / "outputs" / "models"

HORIZONS = list(range(1, 11))
VAL_FOLD_STARTS = [770, 780, 790, 800, 810, 820]
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
    catalog = build_catalog(wide.columns)
    dates = wide["midday_day"].to_list()
    date_to_idx = {d: i for i, d in enumerate(dates)}
    lag0 = ROOT / "work" / "outputs" / "eda" / "05-lag-correlations.csv"
    params = json.loads((MODELS / "lightgbm-tuned-best-params.json").read_text())

    print("Building Fourier-augmented per-horizon matrices + evaluating on clean folds...")
    err = {h: [] for h in HORIZONS}
    for h in HORIZONS:
        th = time.time()
        fm, _ = build_features_for_horizon(wide, catalog, h=h, lag0_pearson_path=lag0, fourier=True)
        fm_idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
        feat = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
        for fs in VAL_FOLD_STARTS:
            test_days = list(range(fs, fs + TEST_M))
            train_mask = fm_idx < (fs - h)
            test_mask = np.isin(fm_idx, test_days)
            Xtr = fm.filter(pl.Series(train_mask)).select(feat)
            ytr = fm.filter(pl.Series(train_mask))["y"].to_numpy()
            Xte = fm.filter(pl.Series(test_mask)).select(feat)
            yte = fm.filter(pl.Series(test_mask))["y"].to_numpy()
            if len(Xtr) < 30 or len(Xte) == 0:
                continue
            model, cmap = lightgbm_train(Xtr, ytr, params=params)
            pred = lightgbm_predict(model, Xte, mapping=cmap)
            err[h].extend(list(yte - pred))
        print(f"  h={h:2d}  ({time.time()-th:.0f}s, +{len(feat)} feats)")

    fb = bucket(err)
    # Compare to non-Fourier (script 11)
    base = json.loads((OUTM / "phase4-honest-eval.json").read_text())["lightgbm_tuned"]["clean_val"]

    dur = time.time() - t0
    summary = {"lightgbm_fourier_clean_val": fb, "lightgbm_nofourier_clean_val": base, "duration_seconds": dur}
    (OUTM / "phase4-fourier.json").write_text(json.dumps(summary, indent=2, default=str))

    md = ["# Phase 4 — Fourier seasonal features (step 3)", "",
          f"Generated: {datetime.now().isoformat(timespec='seconds')} · {dur:.0f}s",
          "Same clean validation folds as script 11; holdout still sealed.", "",
          "Added 10 Fourier terms (3 weekly + 2 annual harmonics × sin/cos) to the calendar block.", "",
          "| LightGBM | overall | 1-5d | 6-10d |", "|---|---:|---:|---:|",
          f"| without Fourier | {base['overall']:.4f} | {base['1-5d']:.4f} | {base['6-10d']:.4f} |",
          f"| with Fourier | {fb['overall']:.4f} | {fb['1-5d']:.4f} | {fb['6-10d']:.4f} |",
          f"| delta | {(base['overall']-fb['overall'])/base['overall']*100:+.1f}% | "
          f"{(base['1-5d']-fb['1-5d'])/base['1-5d']*100:+.1f}% | "
          f"{(base['6-10d']-fb['6-10d'])/base['6-10d']*100:+.1f}% |", ""]
    verdict = "helps" if fb["overall"] < base["overall"] else "does not help"
    md.append(f"Verdict: Fourier **{verdict}** overall on clean validation.")
    (OUTM / "phase4-fourier.md").write_text("\n".join(md))

    print(f"\n=== Fourier ({dur:.0f}s) ===")
    print(f"  without : overall {base['overall']:.4f} | 1-5d {base['1-5d']:.4f} | 6-10d {base['6-10d']:.4f}")
    print(f"  with    : overall {fb['overall']:.4f} | 1-5d {fb['1-5d']:.4f} | 6-10d {fb['6-10d']:.4f}")
    print(f"Report: {OUTM / 'phase4-fourier.md'}")


if __name__ == "__main__":
    main()
