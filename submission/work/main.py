"""main.py — deployment entry-point: produce the official submission forecasts.

FROZEN contest algorithm (team decision 2026-06-05; see DECISIONS.md D5).
At each forecast origin D it RE-FITS both models on all data with an already-
OBSERVED label (decision_day <= D - 3 - h, i.e. respecting the 3-day target
reporting lag and the "only data up to day D" rule), then combines them per
horizon with fixed "weight A":
    day 1-5  : 0.55 * ExtraTrees(calendar+target-history, 39 feats) + 0.45 * LightGBM(full, 1392)
    day 6-10 : 0.90 * ExtraTrees                                    + 0.10 * LightGBM
No future target or future predictor values are ever used (no leakage).

RUN ORDER
    ./run.sh python scripts/01_long_to_wide.py        # build daily wide table
    ./run.sh python scripts/03_feature_catalog.py     # causal-role catalogue
    ./run.sh python scripts/04_feature_engineering.py # per-horizon feature matrices
    ./run.sh python main.py                           # -> outputs/submission/{pred_matrix,mse_summary}.csv

For the assessment: re-run 01/03/04 on the released validation data first (the
development CSV plus the organisers' amended validation CSV, so the feature
matrices span 16 Mar 2023 .. 17 Feb 2026), then run main.py — it forecasts the
131 rolling 10-day periods (1 Oct 2025 .. 17 Feb 2026, the amended assessment
window). On dev-only data it falls back to the most recent valid origins, and
fills mse_summary where the true target is present.

Env: MAX_ORIGINS=<n> limits the number of origins (smoke test). Runtime ~9s/origin.
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
OUT = ROOT / "outputs" / "submission"; OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = list(range(1, 11)); REPORTING_LAG = 3
ET_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
WEIGHT_A = {h: (0.55 if h <= 5 else 0.90) for h in HORIZONS}   # ExtraTrees weight
ASSESS_START, ASSESS_END = "2025-10-01", "2026-03-31"
DUMMY = -9999


def main():
    t0 = time.time()
    wide = load_wide_daily()
    days = wide["midday_day"].to_list()
    d2i = {d: i for i, d in enumerate(days)}
    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    total = wide.shape[0]

    # Pre-load per-horizon matrices + column sets + decision-day index maps
    fms, idxmaps = {}, {}
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        fms[h] = fm
        idxmaps[h] = np.array([d2i[d] for d in fm["decision_day"].to_list()])
    full_cols = [c for c in fms[1].columns if c not in ("decision_day", "label_day", "y")]
    endo_cols = [c for c in full_cols if " " not in c and "(" not in c]

    # Choose forecast origins: the 173 assessment periods if the data reaches them,
    # else the most recent 173 valid dev origins (a demonstration set).
    def in_window(i, lo, hi):
        return lo <= str(days[i]) <= hi
    assess_origins = [i for i in range(total) if i + 10 < total
                      and str(days[i + 1]) >= ASSESS_START and str(days[i + 10]) <= ASSESS_END]
    if assess_origins:
        origins = assess_origins
        mode = "assessment"
    else:
        last_valid = total - 11               # need y(D+10) to exist for scoring
        origins = list(range(max(0, last_valid - 172), last_valid + 1))
        mode = "dev-fallback"
    cap = os.environ.get("MAX_ORIGINS")
    if cap:
        origins = origins[: int(cap)]
    print(f"mode={mode}  origins={len(origins)}  "
          f"({days[origins[0]]} .. {days[origins[-1]]})", flush=True)

    pm_rows, ms_rows = [], []
    for fid, D in enumerate(origins, 1):
        preds = []
        for h in HORIZONS:
            fm, idx = fms[h], idxmaps[h]
            tr = idx <= (D - REPORTING_LAG - h)         # observed-label training rows only
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
        pm_rows.append({"forecast_id": fid, **{f"day_{h}": preds[h - 1] for h in HORIZONS}})
        # MSE only where the true target is present and not dummy
        actual = np.array([y_full[D + h] if D + h < total else np.nan for h in HORIZONS])
        p = np.array(preds)
        def bmse(lo, hi):
            a_, p_ = actual[lo:hi], p[lo:hi]
            m = np.isfinite(a_) & np.isfinite(p_) & (a_ != DUMMY)
            return float(np.mean((a_[m] - p_[m]) ** 2)) if m.sum() else ""
        ms_rows.append({"forecast_id": fid, "mse_1_5": bmse(0, 5), "mse_6_10": bmse(5, 10)})
        if fid % 20 == 0:
            print(f"  {fid}/{len(origins)} origins ({time.time()-t0:.0f}s)", flush=True)

    pl.DataFrame(pm_rows).write_csv(OUT / "pred_matrix.csv")
    pl.DataFrame(ms_rows).write_csv(OUT / "mse_summary.csv")
    # headline (only meaningful where outcomes are real)
    v15 = [r["mse_1_5"] for r in ms_rows if r["mse_1_5"] != ""]
    v610 = [r["mse_6_10"] for r in ms_rows if r["mse_6_10"] != ""]
    if v15:
        print(f"scored {len(v15)} origins: MSE_1-5d={np.mean(v15):.4f}  MSE_6-10d={np.mean(v610):.4f}", flush=True)
    print(f"wrote outputs/submission/pred_matrix.csv ({len(pm_rows)} rows) + mse_summary.csv  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
