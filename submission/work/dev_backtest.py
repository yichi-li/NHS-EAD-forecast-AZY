"""dev_backtest.py — all-in-one development backtest on the most recent N dev
origins, matching the common reporting basis (sdwfrost's report uses "the most
recent 173 origins for which all ten target days exist" on the development data).

Same frozen model as main.py: per origin, re-fit LightGBM(full) + ExtraTrees(endo)
on observed-label rows (idx <= D-3-h), combine with weight A. Restricts origins to
the most recent N dev origins whose 10 target days are all within the development
window (<= DEV_END), so it is directly comparable to other entrants' dev-backtest
figures. Prints MSE_1-5d and MSE_6-10d.
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import load_wide_daily            # noqa: E402
from src.models import lightgbm_train, lightgbm_predict  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.impute import SimpleImputer          # noqa: E402

DATA = ROOT / "data"
HORIZONS = list(range(1, 11)); REPORTING_LAG = 3
ET_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
WEIGHT_A = {h: (0.55 if h <= 5 else 0.90) for h in HORIZONS}
DEV_END = "2025-09-30"
N_BACKTEST = int(os.environ.get("N_BACKTEST", "173"))


def main():
    t0 = time.time()
    wide = load_wide_daily()
    days = wide["midday_day"].to_list()
    d2i = {d: i for i, d in enumerate(days)}
    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    total = wide.shape[0]

    fms, idxmaps = {}, {}
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        fms[h] = fm
        idxmaps[h] = np.array([d2i[d] for d in fm["decision_day"].to_list()])
    full_cols = [c for c in fms[1].columns if c not in ("decision_day", "label_day", "y")]
    endo_cols = [c for c in full_cols if " " not in c and "(" not in c]

    # Most recent N dev origins: day[i+10] must exist and be within the dev window.
    valid = [i for i in range(total) if i + 10 < total and str(days[i + 10]) <= DEV_END]
    origins = valid[-N_BACKTEST:]
    print(f"dev-backtest origins={len(origins)}  "
          f"({days[origins[0]]} .. {days[origins[-1]]}) target up to {days[origins[-1]+10]}", flush=True)

    e15, e610 = [], []
    for fid, D in enumerate(origins, 1):
        preds = []
        for h in HORIZONS:
            fm, idx = fms[h], idxmaps[h]
            tr = idx <= (D - REPORTING_LAG - h)
            te = idx == D
            if tr.sum() < 30 or te.sum() == 0:
                preds.append(np.nan); continue
            Xtr_f = fm.filter(pl.Series(tr)).select(full_cols)
            ytr = fm.filter(pl.Series(tr))["y"].to_numpy()
            Xte_f = fm.filter(pl.Series(te)).select(full_cols)
            Xtr_e = fm.filter(pl.Series(tr)).select(endo_cols).to_numpy()
            Xte_e = fm.filter(pl.Series(te)).select(endo_cols).to_numpy()
            mdl, cm = lightgbm_train(Xtr_f, ytr)
            p_lgb = float(lightgbm_predict(mdl, Xte_f, mapping=cm)[0])
            imp = SimpleImputer(strategy="median")
            a = imp.fit_transform(Xtr_e); b = imp.transform(Xte_e)
            et = ExtraTreesRegressor(**ET_PARAMS); et.fit(a, ytr)
            p_et = float(et.predict(b)[0])
            w = WEIGHT_A[h]
            preds.append(w * p_et + (1 - w) * p_lgb)
        actual = np.array([y_full[D + h] for h in HORIZONS])
        p = np.array(preds)
        for k in range(10):
            if np.isfinite(actual[k]) and np.isfinite(p[k]):
                (e15 if k < 5 else e610).append((actual[k] - p[k]) ** 2)
        if fid % 20 == 0:
            print(f"  {fid}/{len(origins)} ({time.time()-t0:.0f}s)", flush=True)

    m15, m610 = float(np.mean(e15)), float(np.mean(e610))
    print(f"\nDEV-BACKTEST  MSE_1-5d={m15:.4f}  MSE_6-10d={m610:.4f}  "
          f"(n_origins={len(origins)}, {time.time()-t0:.0f}s)", flush=True)
    print("DEVBACKTEST DONE", flush=True)


if __name__ == "__main__":
    main()
