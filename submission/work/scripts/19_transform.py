"""19 — Bonus: does a log1p target transform help ExtraTrees_cal?

ExtraTrees predicts the leaf mean; on a right-skewed, count-like target the high
days dominate. log1p(y) -> fit -> expm1 back-transform can reduce that. Winter y is
higher/more volatile, so the transform might help winter most. Test on winter+summer,
9 folds, per-horizon. Adopt into the freeze ONLY if it clearly beats raw.

OUT: outputs/metrics/19-transform.md/.json
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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUT_M = ROOT / "work" / "outputs" / "metrics"; OUT_M.mkdir(parents=True, exist_ok=True)
HORIZONS = list(range(1, 11)); TEST_M = 10
REGIMES = {"winter2425": (626, 716), "summer25": (830, 920)}
ET = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)

def folds_for(s, e):
    out, k, ts = [], 0, s
    while ts + TEST_M <= e:
        out.append(Fold(k, ts, list(range(ts, ts + TEST_M)))); ts += TEST_M; k += 1
    return out

def run(transform, endo, folds, d2i):
    res = []
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
        for fold in folds:
            trm = train_row_mask_for_fold(idx, fold, h); tem = np.isin(idx, fold.test_decision_days)
            Xtr = fm.filter(pl.Series(trm)).select(endo).to_numpy(); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
            Xte = fm.filter(pl.Series(tem)).select(endo).to_numpy(); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
            if len(Xtr) < 30 or len(Xte) == 0: continue
            imp = SimpleImputer(strategy="median"); a = imp.fit_transform(Xtr); b = imp.transform(Xte)
            yt = np.log1p(ytr) if transform else ytr
            m = ExtraTreesRegressor(**ET); m.fit(a, yt)
            yp = np.expm1(m.predict(b)) if transform else m.predict(b)
            res.append(FoldResult(fold.fold_idx, h, len(yte), mse(yte, yp), yte, yp))
    return summarise_per_horizon(res)

def main():
    t0 = time.time()
    wide = load_wide_daily(); d2i = {d: i for i, d in enumerate(wide["midday_day"].to_list())}
    endo = [c for c in pl.read_parquet(DATA / "feature_matrix_h1.parquet").columns
            if c not in ("decision_day", "label_day", "y") and " " not in c and "(" not in c]
    out = {}
    for regime, (s, e) in REGIMES.items():
        for tr in (False, True):
            try:
                sm = run(tr, endo, folds_for(s, e), d2i)
                out[f"{regime}|{'log1p' if tr else 'raw'}"] = {k: v for k, v in sm.items() if k != "mse_per_h"}
                log(f"  {regime} {'log1p' if tr else 'raw':5s}: overall={sm['mse_overall_mean']:.4f} "
                    f"1-5d={sm['mse_1_5d_mean']:.4f} 6-10d={sm['mse_6_10d_mean']:.4f}")
            except Exception:
                log(f"  !! {regime} tr={tr} FAILED:\n{traceback.format_exc()}")
    out["log"] = LOG
    (OUT_M / "19-transform.json").write_text(json.dumps(out, indent=2, default=str))
    md = ["# Bonus: log1p target transform for ExtraTrees_cal\n",
          f"_Winter + summer, 9 folds, per-horizon. Runtime {time.time()-t0:.0f}s._\n",
          "| regime / target | overall | 1-5d | 6-10d |", "|---|--:|--:|--:|"]
    for regime in REGIMES:
        for tag in ("raw", "log1p"):
            k = f"{regime}|{tag}"
            if k in out:
                v = out[k]
                md.append(f"| {regime} {tag} | {v['mse_overall_mean']:.4f} | {v['mse_1_5d_mean']:.4f} | {v['mse_6_10d_mean']:.4f} |")
    md.append("\n_Adopt log1p into the freeze only if it clearly beats raw (esp. winter)._")
    (OUT_M / "19-transform.md").write_text("\n".join(md))
    log(f"DONE ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
