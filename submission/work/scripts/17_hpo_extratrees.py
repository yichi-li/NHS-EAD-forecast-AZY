"""17 — HPO of ExtraTrees_cal (the winter champion), nested to avoid over-optimism.

Tuning AND reporting on the same folds inflates the result; this uses a nested split:
  - TUNE on winter folds 0-5 (early-mid winter): pick the config with lowest pooled MSE.
  - REPORT that config on HELD-OUT winter folds 6-8 (late winter, tuning never saw them)
    AND on all summer folds. Adopt the tuned config ONLY if it beats the default on the
    held-out winter folds (and doesn't regress badly on summer).

Default = stage-1 ExtraTrees_cal (n_estimators=300, sklearn defaults) → winter 0.1613.
Grid over n_estimators / max_features / min_samples_leaf. ExtraTrees on cal(39) is fast.

OUT: outputs/metrics/17-hpo-extratrees.md / .json
"""
from __future__ import annotations
import json, sys, time, traceback, warnings, itertools
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
LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)

def folds_for(s, e):
    out, k, ts = [], 0, s
    while ts + TEST_M <= e:
        out.append(Fold(k, ts, list(range(ts, ts + TEST_M)))); ts += TEST_M; k += 1
    return out

# cache feature matrices (cal cols) once
_CACHE = {}
def get_fm(h, endo):
    if h not in _CACHE:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        _CACHE[h] = fm
    return _CACHE[h]

def run_config(params, endo, folds, d2i):
    res = []
    for h in HORIZONS:
        fm = get_fm(h, endo)
        idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
        for fold in folds:
            trm = train_row_mask_for_fold(idx, fold, h); tem = np.isin(idx, fold.test_decision_days)
            Xtr = fm.filter(pl.Series(trm)).select(endo).to_numpy(); ytr = fm.filter(pl.Series(trm))["y"].to_numpy()
            Xte = fm.filter(pl.Series(tem)).select(endo).to_numpy(); yte = fm.filter(pl.Series(tem))["y"].to_numpy()
            if len(Xtr) < 30 or len(Xte) == 0: continue
            imp = SimpleImputer(strategy="median")
            a = imp.fit_transform(Xtr); b = imp.transform(Xte)
            m = ExtraTreesRegressor(n_jobs=-1, random_state=42, **params); m.fit(a, ytr)
            res.append(FoldResult(fold.fold_idx, h, len(yte), mse(yte, m.predict(b)), yte, m.predict(b)))
    return summarise_per_horizon(res)

def main():
    t0 = time.time()
    wide = load_wide_daily(); d2i = {d: i for i, d in enumerate(wide["midday_day"].to_list())}
    endo = [c for c in pl.read_parquet(DATA / "feature_matrix_h1.parquet").columns
            if c not in ("decision_day", "label_day", "y") and " " not in c and "(" not in c]
    wfolds = folds_for(*REGIMES["winter2425"]); sfolds = folds_for(*REGIMES["summer25"])
    tune_folds = wfolds[:6]; report_folds = wfolds[6:]
    log(f"cal feats={len(endo)}; tune=winter folds 0-5, report=winter folds 6-8 + summer")

    DEFAULT = {"n_estimators": 300}
    grid = [{"n_estimators": ne, "max_features": mf, "min_samples_leaf": ml}
            for ne, mf, ml in itertools.product([300, 600, 1000], ["sqrt", "log2", 0.5, 1.0], [1, 2, 3])]

    # tune
    tuned = []
    for i, p in enumerate(grid):
        try:
            sm = run_config(p, endo, tune_folds, d2i)
            tuned.append((p, sm["mse_overall_mean"]))
            if i % 6 == 0: log(f"  trial {i+1}/{len(grid)} {p} -> tune MSE {sm['mse_overall_mean']:.4f}")
        except Exception:
            log(f"  !! config {p} FAILED:\n{traceback.format_exc()}")
    tuned.sort(key=lambda x: x[1])
    best_p = tuned[0][0] if tuned else DEFAULT
    log(f"best on tune folds: {best_p} (tune MSE {tuned[0][1]:.4f})")

    # honest report on held-out winter folds 6-8 + summer, default vs best
    out = {"grid_size": len(grid), "best_params": best_p, "default": DEFAULT, "report": {}}
    for name, params in [("default", DEFAULT), ("tuned", best_p)]:
        rw = run_config(params, endo, report_folds, d2i)
        rs = run_config(params, endo, sfolds, d2i)
        out["report"][name] = {
            "winter_heldout_overall": rw["mse_overall_mean"], "winter_heldout_1_5d": rw["mse_1_5d_mean"],
            "winter_heldout_6_10d": rw["mse_6_10d_mean"], "summer_overall": rs["mse_overall_mean"]}
        log(f"  REPORT {name}: winter-heldout {rw['mse_overall_mean']:.4f} | summer {rs['mse_overall_mean']:.4f}")

    d = out["report"]["default"]; t = out["report"]["tuned"]
    adopt = (t["winter_heldout_overall"] <= d["winter_heldout_overall"] - 1e-4) and \
            (t["summer_overall"] <= d["summer_overall"] + 0.003)
    out["adopt_tuned"] = bool(adopt); out["log"] = LOG
    (OUT_M / "17-hpo-extratrees.json").write_text(json.dumps(out, indent=2, default=str))

    md = ["# HPO ExtraTrees_cal (nested: tune folds 0-5, report folds 6-8 + summer)\n",
          f"_Grid {len(grid)} configs. Runtime {time.time()-t0:.0f}s._\n",
          f"**Best params (on tune folds):** `{best_p}`\n",
          "| config | winter-heldout overall | 1-5d | 6-10d | summer overall |",
          "|---|--:|--:|--:|--:|",
          f"| default (n_est=300) | {d['winter_heldout_overall']:.4f} | {d['winter_heldout_1_5d']:.4f} | {d['winter_heldout_6_10d']:.4f} | {d['summer_overall']:.4f} |",
          f"| tuned | {t['winter_heldout_overall']:.4f} | {t['winter_heldout_1_5d']:.4f} | {t['winter_heldout_6_10d']:.4f} | {t['summer_overall']:.4f} |",
          f"\n**Adopt tuned? {'YES' if adopt else 'NO — keep default'}** "
          f"(tuned must beat default on held-out winter without regressing summer).\n",
          "_Note: held-out report = 3 winter folds (small sample, noisy); adoption is conservative._"]
    (OUT_M / "17-hpo-extratrees.md").write_text("\n".join(md))
    log(f"DONE adopt={adopt} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
