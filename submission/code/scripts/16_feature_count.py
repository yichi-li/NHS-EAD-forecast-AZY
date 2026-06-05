"""16 — Feature-count ablation: cal(39) + top-K metrics. Does a sweet spot beat cal?

Stage-1 found ExtraTrees_cal (39 endogenous feats) beats ExtraTrees_full (1383):
the raw 220-metric columns add net NOISE for ExtraTrees. Question: does adding back
only the K STRONGEST metrics (less noise, keep signal) beat cal(39)?

- Rank exogenous metric columns by |Pearson corr with y| on the pre-winter training
  pool (decision_day < winter start). Deterministic.
- For K in {0,10,20,50,100}: featset = endo(39) + topK exo. K=0 == cal (sanity check
  vs stage 1). Run ExtraTrees + LightGBM, winter + summer, 9 folds, per-horizon.
- Goal: find the feature set with lowest WINTER MSE (the scored regime).

Robust (try/except per run), full auto tables. OUT: outputs/metrics/16-feature-count.md/.json
"""
from __future__ import annotations
import json, sys, time, traceback, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import Fold, FoldResult, mse, summarise_per_horizon, train_row_mask_for_fold
from src.data import load_wide_daily
from src.models import lightgbm_train, lightgbm_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUT_M = ROOT / "work" / "outputs" / "metrics"; OUT_M.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11)); TEST_M = 10
REGIMES = {"winter2425": (626, 716), "summer25": (830, 920)}
KS = [0, 10, 20, 50, 100]
ET_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)

def folds_for(s, e):
    out, k, ts = [], 0, s
    while ts + TEST_M <= e:
        out.append(Fold(k, ts, list(range(ts, ts + TEST_M)))); ts += TEST_M; k += 1
    return out

def fit_pred(kind, Xtr, ytr, Xte):
    if kind == "lightgbm":
        m, cm = lightgbm_train(Xtr, ytr); return lightgbm_predict(m, Xte, mapping=cm)
    imp = SimpleImputer(strategy="median")
    a = imp.fit_transform(Xtr.to_numpy()); b = imp.transform(Xte.to_numpy())
    m = ExtraTreesRegressor(**ET_PARAMS); m.fit(a, ytr); return m.predict(b)

def rank_exo(d2i):
    fm = pl.read_parquet(DATA / "feature_matrix_h1.parquet")
    cols = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
    endo = [c for c in cols if (" " not in c and "(" not in c)]
    exo = [c for c in cols if c not in endo]
    idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
    tr = idx < REGIMES["winter2425"][0]           # pre-winter training pool
    y = fm.filter(pl.Series(tr))["y"].to_numpy()
    scores = {}
    sub = fm.filter(pl.Series(tr))
    for c in exo:
        x = sub[c].to_numpy().astype(float)
        m = ~(np.isnan(x) | np.isnan(y))
        scores[c] = abs(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 10 and np.nanstd(x[m]) > 0 else 0.0
    ranked = sorted(exo, key=lambda c: scores.get(c, 0.0), reverse=True)
    log(f"ranked {len(exo)} exo metrics; top5: {[ (c[:30], round(scores[c],2)) for c in ranked[:5] ]}")
    return endo, ranked

def run(kind, fc, folds, d2i):
    res = []
    for h in HORIZONS:
        try:
            fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
            idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
            for fold in folds:
                trm = train_row_mask_for_fold(idx, fold, h); tem = np.isin(idx, fold.test_decision_days)
                Xtr = fm.filter(pl.Series(trm)).select(fc); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
                Xte = fm.filter(pl.Series(tem)).select(fc); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
                if len(Xtr) < 30 or len(Xte) == 0: continue
                yp = fit_pred(kind, Xtr, ytr, Xte)
                res.append(FoldResult(fold.fold_idx, h, len(yte), mse(yte, yp), yte, np.asarray(yp)))
        except Exception:
            log(f"  !! {kind} h={h} FAILED:\n{traceback.format_exc()}")
    return summarise_per_horizon(res)

def main():
    t0 = time.time()
    wide = load_wide_daily(); days = wide["midday_day"].to_list()
    d2i = {d: i for i, d in enumerate(days)}
    endo, ranked = rank_exo(d2i)
    results = {}   # (kind, K, regime) -> summary
    for K in KS:
        fc = endo + ranked[:K]
        log(f"=== K={K}  ({len(fc)} features) ===")
        for kind in ("extratrees", "lightgbm"):
            for regime, (s, e) in REGIMES.items():
                try:
                    sm = run(kind, fc, folds_for(s, e), d2i)
                    results[f"{kind}|K{K}|{regime}"] = sm
                    log(f"  {kind} K={K} {regime}: overall={sm['mse_overall_mean']:.4f} "
                        f"1-5d={sm['mse_1_5d_mean']:.4f} 6-10d={sm['mse_6_10d_mean']:.4f}")
                except Exception:
                    log(f"  !! {kind} K={K} {regime} FAILED:\n{traceback.format_exc()}")
        _dump(results)
    _report(results, time.time() - t0)
    log(f"DONE ({time.time()-t0:.0f}s)")

def _dump(results):
    (OUT_M / "16-feature-count.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "mse_per_h"} for k, v in results.items()}
        | {"log": LOG}, indent=2, default=str))

def _report(results, secs):
    md = ["# Feature-count ablation — cal(39) + top-K metrics\n",
          f"_K=0 is cal (sanity-check vs stage 1). Winter is the scored regime. Runtime {secs:.0f}s._\n",
          "_Reference (stage 1, full=1383): ExtraTrees_full winter 0.1700; LightGBM_full winter 0.1815._\n"]
    for kind in ("extratrees", "lightgbm"):
        md.append(f"\n## {kind}\n")
        md.append("| K (feats) | winter overall | winter 1-5d | winter 6-10d | summer overall |")
        md.append("|--:|--:|--:|--:|--:|")
        for K in KS:
            w = results.get(f"{kind}|K{K}|winter2425"); s = results.get(f"{kind}|K{K}|summer25")
            if not w: continue
            nf = 39 + K
            md.append(f"| {K} ({nf}) | {w['mse_overall_mean']:.4f} | {w['mse_1_5d_mean']:.4f} | "
                      f"{w['mse_6_10d_mean']:.4f} | {s['mse_overall_mean']:.4f} |" if s else
                      f"| {K} ({nf}) | {w['mse_overall_mean']:.4f} | {w['mse_1_5d_mean']:.4f} | {w['mse_6_10d_mean']:.4f} | - |")
    md.append("\n_Lower winter MSE = better. Compare to ExtraTrees_cal=0.1613 (K=0).Find the sweet spot._")
    (OUT_M / "16-feature-count.md").write_text("\n".join(md))

if __name__ == "__main__":
    main()
