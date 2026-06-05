"""13 — Phase 4 final: best-config (Fourier LightGBM + per-horizon ensemble)
confirmed on the SEALED holdout, with a full attribution table.

Protocol (clean):
  1. Build Fourier-augmented per-horizon matrices (kept in memory).
  2. Clean validation (folds 770-829, HPO-uncontaminated): train Fourier-LGB,
     compute sNaive/ES, SELECT per-horizon ensemble weights here.
  3. Holdout (870-929, sealed, opened once): train Fourier-LGB on all CV-pool
     data, predict, apply the selected weights → final honest numbers.

Outputs: outputs/wave3/PHASE4-REPORT.md / .json + forecasts/phase4-holdout-pred.csv
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
from src.cv import HoldoutSplit
from src.models import lightgbm_train, lightgbm_predict, snaive_forecast, es_forecast

ROOT = Path(__file__).resolve().parents[2]
OUTM = ROOT / "work" / "outputs" / "metrics"
OUTF = ROOT / "work" / "outputs" / "forecasts"
OUTW = ROOT / "work" / "outputs" / "wave3"
MODELS = ROOT / "work" / "outputs" / "models"
OUTW.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
REPORTING_LAG = 3
# 4 HPO-uncontaminated folds (all < 830). Reduced from 6 to keep the run well
# under any background-task time limit; still enough to pick low-dim weights.
VAL_FOLD_STARTS = [790, 800, 810, 820]
TEST_M = 10


def bucket(errs_by_h):
    out = {}
    for lab, lo, hi in [("overall", 1, 10), ("1-5d", 1, 5), ("6-10d", 6, 10)]:
        vals = [e for h in range(lo, hi + 1) for e in errs_by_h.get(h, [])]
        out[lab] = float(np.mean(np.square(vals))) if vals else float("nan")
    return out


def naive_preds(y_full, Di, h):
    obs = y_full[: Di - (REPORTING_LAG - 1)]
    sn = float(snaive_forecast(obs, horizon=10, start_offset=REPORTING_LAG + 1)[h - 1])
    try:
        es = float(es_forecast(obs, horizon=10, start_offset=REPORTING_LAG + 1)[h - 1])
    except Exception:
        es = float(np.nanmean(obs))
    return sn, es


def main():
    t0 = time.time()
    wide = load_wide_daily()
    catalog = build_catalog(wide.columns)
    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    dates = wide["midday_day"].to_list()
    d2i = {d: i for i, d in enumerate(dates)}
    total = wide.shape[0]
    cv_slice, _ = HoldoutSplit().split(total)
    cv_end = cv_slice.stop
    params = json.loads((MODELS / "lightgbm-tuned-best-params.json").read_text())
    lag0 = ROOT / "work" / "outputs" / "eda" / "05-lag-correlations.csv"
    ho_days = list(range(cv_end, total - max(HORIZONS)))

    print(f"Building Fourier matrices + clean-val weight selection + holdout confirm")
    print(f"Holdout: {dates[ho_days[0]]} .. {dates[ho_days[-1]]} ({len(ho_days)} decision days)\n")

    # validation prediction store, and holdout error store
    valrows = []
    ho_lgb = {h: [] for h in HORIZONS}; ho_sn = {h: [] for h in HORIZONS}
    ho_es = {h: [] for h in HORIZONS}; ho_en = {h: [] for h in HORIZONS}
    weights = {}
    ho_rows = []

    for h in HORIZONS:
        th = time.time()
        fm, _ = build_features_for_horizon(wide, catalog, h=h, lag0_pearson_path=lag0, fourier=True)
        fm_idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
        feat = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]

        # ----- clean validation: collect predictions for weight selection -----
        vsub = []
        for fs in VAL_FOLD_STARTS:
            td = list(range(fs, fs + TEST_M))
            trm = fm_idx < (fs - h); tem = np.isin(fm_idx, td)
            Xtr = fm.filter(pl.Series(trm)).select(feat); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
            Xte = fm.filter(pl.Series(tem)).select(feat); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
            dte = fm.filter(pl.Series(tem))["decision_day"].to_list()
            if len(Xtr) < 30 or len(Xte) == 0:
                continue
            mdl, cm = lightgbm_train(Xtr, ytr, params=params)
            pr = lightgbm_predict(mdl, Xte, mapping=cm)
            for D, yt, p in zip(dte, yte, pr):
                sn, es = naive_preds(y_full, d2i[D], h)
                vsub.append((yt, p, sn, es))
        vsub = np.array(vsub)  # (n,4): y, lgb, sn, es
        yt, L, S, E = vsub[:, 0], vsub[:, 1], vsub[:, 2], vsub[:, 3]
        # grid search 3-way weights
        grid = np.linspace(0, 1, 11)
        best, best_mse = (1.0, 0.0, 0.0), float(np.mean((yt - L) ** 2))
        for wl in grid:
            for ws in grid:
                we = round(1 - wl - ws, 4)
                if we < -1e-9:
                    continue
                m = float(np.mean((yt - (wl * L + ws * S + we * E)) ** 2))
                if m < best_mse:
                    best_mse, best = m, (float(wl), float(ws), float(we))
        weights[h] = {"w_lgb": best[0], "w_snaive": best[1], "w_es": best[2]}

        # ----- holdout: train on all CV-pool data, predict, apply weights -----
        trm = fm_idx < (cv_end - h + 1); tem = np.isin(fm_idx, ho_days)
        Xtr = fm.filter(pl.Series(trm)).select(feat); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
        Xte = fm.filter(pl.Series(tem)).select(feat); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
        dte = fm.filter(pl.Series(tem))["decision_day"].to_list()
        mdl, cm = lightgbm_train(Xtr, ytr, params=params)
        pr = lightgbm_predict(mdl, Xte, mapping=cm)
        w = weights[h]
        for D, ytv, p in zip(dte, yte, pr):
            sn, es = naive_preds(y_full, d2i[D], h)
            blend = w["w_lgb"] * p + w["w_snaive"] * sn + w["w_es"] * es
            ho_lgb[h].append(ytv - p); ho_sn[h].append(ytv - sn)
            ho_es[h].append(ytv - es); ho_en[h].append(ytv - blend)
            ho_rows.append({"h": h, "decision_day": str(D), "y_true": float(ytv),
                            "lgb_fourier": float(p), "snaive": sn, "es": es, "ensemble": float(blend)})
        print(f"  h={h:2d}  weights LGB/sN/ES={w['w_lgb']:.1f}/{w['w_snaive']:.1f}/{w['w_es']:.1f}  ({time.time()-th:.0f}s)")

    pl.DataFrame(ho_rows).write_csv(OUTF / "phase4-holdout-pred.csv")
    L = bucket(ho_lgb); S = bucket(ho_sn); E = bucket(ho_es); EN = bucket(ho_en)
    dur = time.time() - t0

    out = {"holdout_period": f"{dates[ho_days[0]]} .. {dates[total-1]}",
           "config": "Fourier-LightGBM (tuned) + per-horizon ensemble with sNaive/ES",
           "snaive": S, "es": E, "lightgbm_fourier": L, "ensemble": EN,
           "per_horizon_weights": weights, "duration_seconds": dur}
    (OUTW / "PHASE4-REPORT.json").write_text(json.dumps(out, indent=2, default=str))

    md = ["# Phase 4 — final result on the sealed holdout", "",
          f"Generated: {datetime.now().isoformat(timespec='seconds')} · {dur:.0f}s",
          f"Holdout: **{out['holdout_period']}** ({len(ho_days)} decision days × 10 h), opened once.",
          "Config: **Fourier-LightGBM (tuned) + per-horizon ensemble (weights chosen on clean validation, not holdout).**", "",
          "## Final holdout MSE", "",
          "| Method | overall | 1-5d | 6-10d |", "|---|---:|---:|---:|",
          f"| sNaive | {S['overall']:.4f} | {S['1-5d']:.4f} | {S['6-10d']:.4f} |",
          f"| ES | {E['overall']:.4f} | {E['1-5d']:.4f} | {E['6-10d']:.4f} |",
          f"| LightGBM + Fourier | {L['overall']:.4f} | {L['1-5d']:.4f} | {L['6-10d']:.4f} |",
          f"| **Ensemble (final)** | **{EN['overall']:.4f}** | **{EN['1-5d']:.4f}** | **{EN['6-10d']:.4f}** |", "",
          "## Honest end-to-end attribution (holdout MSE_overall)", "",
          "| stage | overall | 1-5d | 6-10d | note |", "|---|---:|---:|---:|---|",
          "| v1 LightGBM (LEAKY) | 0.0210 | 0.0157 | 0.0264 | inflated by leakage — not real |",
          "| v2.1 LightGBM default | 0.0506 | 0.0396 | 0.0615 | leakage fixed |",
          "| HPO-tuned (no Fourier) | 0.0506 | 0.0396 | 0.0615 | HPO CV was optimistic (~0.036); honest holdout ~0.051 |",
          f"| + Fourier features | {L['overall']:.4f} | {L['1-5d']:.4f} | {L['6-10d']:.4f} | cyclic weekly+annual encoding |",
          f"| **+ ensemble (final)** | **{EN['overall']:.4f}** | **{EN['1-5d']:.4f}** | **{EN['6-10d']:.4f}** | LGB+sNaive+ES per-h |", "",
          f"**Final ensemble vs sNaive floor: overall {(S['overall']-EN['overall'])/S['overall']*100:+.1f}%, "
          f"1-5d {(S['1-5d']-EN['1-5d'])/S['1-5d']*100:+.1f}%, 6-10d {(S['6-10d']-EN['6-10d'])/S['6-10d']*100:+.1f}%.**", "",
          "## Per-horizon ensemble weights (clean validation)", "",
          "| h | w_LGB | w_sNaive | w_ES |", "|---:|---:|---:|---:|"]
    for h in HORIZONS:
        w = weights[h]
        md.append(f"| {h} | {w['w_lgb']:.1f} | {w['w_snaive']:.1f} | {w['w_es']:.1f} |")
    (OUTW / "PHASE4-REPORT.md").write_text("\n".join(md))

    print(f"\n=== FINAL HOLDOUT ({dur:.0f}s) ===")
    print(f"  sNaive            : overall {S['overall']:.4f} | 1-5d {S['1-5d']:.4f} | 6-10d {S['6-10d']:.4f}")
    print(f"  LightGBM+Fourier  : overall {L['overall']:.4f} | 1-5d {L['1-5d']:.4f} | 6-10d {L['6-10d']:.4f}")
    print(f"  Ensemble (FINAL)  : overall {EN['overall']:.4f} | 1-5d {EN['1-5d']:.4f} | 6-10d {EN['6-10d']:.4f}")
    print(f"Report: {OUTW / 'PHASE4-REPORT.md'}")


if __name__ == "__main__":
    main()
