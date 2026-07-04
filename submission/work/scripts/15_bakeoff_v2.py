"""15 — Bake-off v2: the freeze decision. Robust, full auto-generated tables.

Model roster:
  - DROP CatBoost (slowest, not best) and Zixuan_actual (he may submit separately).
  - 6 single models = {LightGBM, XGBoost, ExtraTrees} x {full 1383 feats, cal 39 feats}.
  - sNaive reference.
  - 2 regimes (drop winter-2023-24, training pool too short):
        winter2425 = 2024-12-01..2025-02-28  (idx 626..715)
        summer25   = ~2025-06-23..2025-09-20 (idx 830..919)   [last usable summer window]
    Each regime = 9 expanding-window folds, test_m=10, leakage-safe (src.cv).

Ensemble (FIXED weights here; adaptive lives in the deployment framework, script 18):
  - residual-correlation matrix (diversity diagnostic)
  - greedy ensemble selection per horizon (Caruana-style, overfit-resistant)
  - inverse-MSE(top-3) and equal-weight(top-2) comparators

Seed robustness: re-run the 3 strongest model classes on winter with 2 extra seeds.

Everything wrapped in try/except so one failure never kills the batch. Outputs
written incrementally. Full auto-generated tables (no hand-picking).

OUT:
  outputs/metrics/15-bakeoff-v2.md / .json
  outputs/forecasts/15-bakeoff-v2-pred.csv
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
from src.models import snaive_forecast, lightgbm_train, lightgbm_predict
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUT_M = ROOT / "work" / "outputs" / "metrics"
OUT_F = ROOT / "work" / "outputs" / "forecasts"
for p in (OUT_M, OUT_F): p.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11))
REPORTING_LAG = 3
TEST_M = 10
REGIMES = {"winter2425": (626, 716), "summer25": (830, 920)}

XGB_PARAMS = {"objective": "reg:tweedie", "tweedie_variance_power": 1.1, "n_estimators": 1500,
              "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 100, "subsample": 0.8,
              "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0}
ET_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}

LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(s, flush=True); LOG.append(s)


def folds_for(start, end):
    out, k, ts = [], 0, start
    while ts + TEST_M <= end:
        out.append(Fold(k, ts, list(range(ts, ts + TEST_M)))); ts += TEST_M; k += 1
    return out


def fit_predict(kind, Xtr, ytr, Xte, seed=42):
    if kind == "lightgbm":
        m, cm = lightgbm_train(Xtr, ytr); return lightgbm_predict(m, Xte, mapping=cm)
    if kind == "xgboost":
        p = dict(XGB_PARAMS); p["random_state"] = seed
        m = xgb.XGBRegressor(**p); m.fit(Xtr.to_pandas(), ytr, verbose=False); return m.predict(Xte.to_pandas())
    if kind == "extratrees":
        imp = SimpleImputer(strategy="median")
        a = imp.fit_transform(Xtr.to_numpy()); b = imp.transform(Xte.to_numpy())
        p = dict(ET_PARAMS); p["random_state"] = seed
        m = ExtraTreesRegressor(**p); m.fit(a, ytr); return m.predict(b)
    raise ValueError(kind)


def run_single(regime, label, kind, featcols_by_h, folds, date_to_idx, wide, store, seed=42):
    res = []
    t0 = time.time()
    for h in HORIZONS:
        try:
            fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
            idx = np.array([date_to_idx[d] for d in fm["decision_day"].to_list()])
            fc = featcols_by_h[h]
            for fold in folds:
                trm = train_row_mask_for_fold(idx, fold, h)
                tem = np.isin(idx, fold.test_decision_days)
                Xtr = fm.filter(pl.Series(trm)).select(fc); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
                Xte = fm.filter(pl.Series(tem)).select(fc); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
                dts = fm.filter(pl.Series(tem))["decision_day"].to_list()
                if len(Xtr) < 30 or len(Xte) == 0: continue
                yp = fit_predict(kind, Xtr, ytr, Xte, seed=seed)
                res.append(FoldResult(fold.fold_idx, h, len(yte), mse(yte, yp), yte, np.asarray(yp)))
                if seed == 42:
                    for D, yt, pp in zip(dts, yte, yp):
                        store.setdefault((h, str(D)), {"y_true": float(yt)})[label] = float(pp)
        except Exception:
            log(f"  !! {label} h={h} FAILED:\n{traceback.format_exc()}")
    log(f"  {regime}/{label} done ({time.time()-t0:.0f}s, {len(res)} fold-results)")
    return summarise_per_horizon(res)


def run_snaive(regime, folds, y_full, total, wide, store):
    res = []
    for fold in folds:
        for D in fold.test_decision_days:
            obs = y_full[: D - (REPORTING_LAG - 1)]
            try: p = snaive_forecast(obs, horizon=10, start_offset=REPORTING_LAG + 1)
            except Exception: p = np.full(10, np.nanmean(obs))
            for hi, h in enumerate(HORIZONS):
                t = D + h
                if t >= total: continue
                yt, yp = float(y_full[t]), float(p[hi])
                res.append(FoldResult(fold.fold_idx, h, 1, (yt - yp) ** 2, np.array([yt]), np.array([yp])))
                store.setdefault((h, str(wide["midday_day"][D])), {"y_true": yt})["sNaive"] = yp
    return summarise_per_horizon(res)


# ---------- ensembles (fixed) from a regime's prediction store ----------
def aligned(store, h, models):
    keys = [k for k in store if k[0] == h and all(m in store[k] for m in models)]
    if not keys: return None, None
    y = np.array([store[k]["y_true"] for k in keys])
    P = {m: np.array([store[k][m] for k in keys]) for m in models}
    return y, P


def greedy_ensemble(y, P, max_picks=12):
    models = list(P); pick = []; best = np.inf
    for _ in range(max_picks):
        cb, cm = None, np.inf
        for m in models:
            pred = np.mean([P[x] for x in pick + [m]], axis=0)
            e = float(np.mean((y - pred) ** 2))
            if e < cm: cm, cb = e, m
        if cm < best - 1e-12: pick.append(cb); best = cm
        else: break
    w = {m: pick.count(m) / len(pick) for m in set(pick)} if pick else {}
    return w, best


def inv_mse_topk(y, P, mse_by_model, k=3):
    top = sorted(mse_by_model, key=mse_by_model.get)[:k]
    inv = {m: 1.0 / mse_by_model[m] for m in top if mse_by_model[m] > 0}
    s = sum(inv.values()) or 1.0
    w = {m: inv[m] / s for m in inv}
    pred = sum(w[m] * P[m] for m in w)
    return w, float(np.mean((y - pred) ** 2))


def equal_top2(y, P, mse_by_model):
    top = sorted(mse_by_model, key=mse_by_model.get)[:2]
    pred = np.mean([P[m] for m in top], axis=0)
    return top, float(np.mean((y - pred) ** 2))


def main():
    t0 = time.time()
    wide = load_wide_daily()
    days = wide["midday_day"].to_list()
    d2i = {d: i for i, d in enumerate(days)}
    y_full = wide["estimated_avoidable_deaths"].to_numpy()
    total = wide.shape[0]

    full_cols, endo_cols = {}, {}
    for h in HORIZONS:
        cols = [c for c in pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet").columns
                if c not in ("decision_day", "label_day", "y")]
        full_cols[h] = cols
        endo_cols[h] = [c for c in cols if (" " not in c and "(" not in c)]
    log(f"features: full={len(full_cols[1])} cal={len(endo_cols[1])}")

    MODELS = [("LightGBM_full", "lightgbm", full_cols), ("LightGBM_cal", "lightgbm", endo_cols),
              ("XGBoost_full", "xgboost", full_cols), ("XGBoost_cal", "xgboost", endo_cols),
              ("ExtraTrees_full", "extratrees", full_cols), ("ExtraTrees_cal", "extratrees", endo_cols)]
    SINGLE_LABELS = ["sNaive"] + [m[0] for m in MODELS]

    summaries = {}   # regime -> label -> summary
    stores = {}      # regime -> pred store
    for regime, (s, e) in REGIMES.items():
        log(f"=== REGIME {regime}  idx[{s},{e})  {days[s]}..{days[e-1]} ===")
        folds = folds_for(s, e)
        store = {}; summaries[regime] = {}
        summaries[regime]["sNaive"] = run_snaive(regime, folds, y_full, total, wide, store)
        for label, kind, fc in MODELS:
            summaries[regime][label] = run_single(regime, label, kind, fc, folds, d2i, wide, store)
        stores[regime] = store
        # write partial json after each regime so we never lose progress
        _dump_json(summaries)

    # ---------- ensembles per regime ----------
    ens = {}  # regime -> dict
    for regime in REGIMES:
        try:
            store = stores[regime]
            models = [m[0] for m in MODELS]
            per = {"greedy": {}, "invmse3": {}, "equal2": {}, "weights_greedy": {}}
            for h in HORIZONS:
                y, P = aligned(store, h, models)
                if y is None: continue
                msebymodel = {m: float(np.mean((y - P[m]) ** 2)) for m in models}
                wg, eg = greedy_ensemble(y, P); per["greedy"][h] = eg; per["weights_greedy"][h] = wg
                _, ei = inv_mse_topk(y, P, msebymodel); per["invmse3"][h] = ei
                _, ee = equal_top2(y, P, msebymodel); per["equal2"][h] = ee
            ens[regime] = per
        except Exception:
            log(f"ensemble {regime} FAILED:\n{traceback.format_exc()}")

    # ---------- residual correlation (winter, pooled over horizons) ----------
    rescorr = None
    try:
        store = stores["winter2425"]; models = [m[0] for m in MODELS]
        rows = {m: [] for m in models}
        for k in store:
            if all(m in store[k] for m in models):
                for m in models: rows[m].append(store[k]["y_true"] - store[k][m])
        R = np.array([rows[m] for m in models])
        rescorr = {"models": models, "matrix": np.corrcoef(R).round(2).tolist()}
    except Exception:
        log(f"rescorr FAILED:\n{traceback.format_exc()}")

    # ---------- seed robustness (winter) ----------
    seedrob = {}
    try:
        folds = folds_for(*REGIMES["winter2425"])
        for label, kind, fc in [("ExtraTrees_cal", "extratrees", endo_cols),
                                 ("ExtraTrees_full", "extratrees", full_cols),
                                 ("LightGBM_full", "lightgbm", full_cols)]:
            vals = []
            for seed in (42, 7, 123):
                sm = run_single("seedchk", f"{label}_s{seed}", kind, fc, folds, d2i, wide, {}, seed=seed)
                vals.append(sm["mse_overall_mean"])
            seedrob[label] = {"seeds": [42, 7, 123], "overall_mse": [round(v, 4) for v in vals],
                              "range": round(max(vals) - min(vals), 4)}
            log(f"  seed-robustness {label}: {seedrob[label]}")
    except Exception:
        log(f"seed robustness FAILED:\n{traceback.format_exc()}")

    _write_report(summaries, ens, rescorr, seedrob, days, time.time() - t0)
    _dump_json(summaries, ens, rescorr, seedrob)
    # flat predictions
    try:
        recs = []
        for regime in REGIMES:
            for (h, D), pr in sorted(stores[regime].items()):
                r = {"regime": regime, "h": h, "decision_day": D}; r.update(pr); recs.append(r)
        pl.DataFrame(recs).write_csv(OUT_F / "15-bakeoff-v2-pred.csv")
    except Exception:
        log(f"pred csv FAILED:\n{traceback.format_exc()}")
    log(f"ALL DONE ({time.time()-t0:.0f}s)")


def _dump_json(summaries, ens=None, rescorr=None, seedrob=None):
    obj = {"summaries": {r: {l: {k: v for k, v in s.items() if k != "mse_per_h"}
                             for l, s in d.items()} for r, d in summaries.items()}}
    if ens is not None: obj["ensemble"] = ens
    if rescorr is not None: obj["residual_corr"] = rescorr
    if seedrob is not None: obj["seed_robustness"] = seedrob
    obj["log"] = LOG
    (OUT_M / "15-bakeoff-v2.json").write_text(json.dumps(obj, indent=2, default=str))


def _row(s):
    return f"| {s['mse_overall_mean']:.4f} | {s['mse_1_5d_mean']:.4f} | {s['mse_6_10d_mean']:.4f} | {s.get('mse_overall_std', float('nan')):.4f} |"


def _write_report(summaries, ens, rescorr, seedrob, days, secs):
    md = ["# Bake-off v2 — freeze decision (winter + summer, all leakage-safe)\n"]
    md.append(f"_Regimes: winter2425 ({days[626]}..{days[715]}), summer25 ({days[830]}..{days[919]}); "
              f"9 expanding folds each. Runtime {secs:.0f}s._\n")
    order = ["sNaive", "LightGBM_full", "LightGBM_cal", "XGBoost_full", "XGBoost_cal",
             "ExtraTrees_full", "ExtraTrees_cal"]
    for regime in summaries:
        md.append(f"\n## {regime} — single models (out-of-sample)\n")
        md.append("| Model | overall | 1-5d | 6-10d | std |\n|---|--:|--:|--:|--:|")
        for l in order:
            if l in summaries[regime]: md.append(f"| {l} " + _row(summaries[regime][l]))
        if regime in ens:
            md.append(f"\n### {regime} — ensemble (fixed weights), per-horizon MSE\n")
            md.append("| h | greedy | invMSE-top3 | equal-top2 | best single |\n|--:|--:|--:|--:|--:|")
            for h in HORIZONS:
                g = ens[regime]["greedy"].get(h, float('nan'))
                i = ens[regime]["invmse3"].get(h, float('nan'))
                e2 = ens[regime]["equal2"].get(h, float('nan'))
                bs = min((summaries[regime][l]["mse_per_h"].get(h, {}).get("mean", float('inf'))
                          for l in order if l != "sNaive"), default=float('nan'))
                md.append(f"| {h} | {g:.4f} | {i:.4f} | {e2:.4f} | {bs:.4f} |")
    if rescorr:
        md.append("\n## Residual correlation (winter, pooled) — lower = more complementary\n")
        ms = rescorr["models"]; md.append("| | " + " | ".join(ms) + " |")
        md.append("|" + "---|" * (len(ms) + 1))
        for i, m in enumerate(ms):
            md.append(f"| {m} | " + " | ".join(f"{rescorr['matrix'][i][j]:.2f}" for j in range(len(ms))) + " |")
    if seedrob:
        md.append("\n## Seed robustness (winter, overall MSE across seeds 42/7/123)\n")
        md.append("| Model | seed MSEs | range |\n|---|---|--:|")
        for m, d in seedrob.items():
            md.append(f"| {m} | {d['overall_mse']} | {d['range']} |")
    md.append("\n## Notes\n- Adaptive ensemble weights are deferred to the deployment framework (script 18).")
    md.append("- `_cal` = 39 endogenous features (y-history + calendar + 6 causal-aggregates); `_full` = 1383.")
    (OUT_M / "15-bakeoff-v2.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
