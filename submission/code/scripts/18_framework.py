"""18 — Deployment framework + dev dry-run (the thing we freeze & submit).

Implements the FROZEN code exactly as it will run on the assessment set, then
dry-runs it over the dev winter window to get an honest contest-metric number and
to emit the official submission format.

Per the rules (README):
  - At each origin D, forecast y(D+1..D+10) using ONLY data up to midday D.
  - Target y has a 3-day reporting lag: training pairs (d, y(d+h)) are usable only
    if y(d+h) is observed at D, i.e. d <= D - 3 - h.  (This is STRICTER than the
    fold-validation mask in script 15, which allowed labels up to D-1; the gap is
    ~2 labels and does not change model rankings, but here we are deployment-honest.)
  - Recalibration is allowed (L74): we RE-FIT both models per origin on the
    expanding window. <1h per 10-day forecast set (we use ~9s).

Models (frozen): ExtraTrees_cal (39 endogenous feats) + LightGBM_full (1383 feats)
— the two most complementary strong models from the bake-off.

We fit both ONCE per (origin, h), store their predictions, then compare combination
rules post-hoc (no refit needed):
  - pure_ET, pure_LGB
  - fixed blends w*ET + (1-w)*LGB for a grid of w (per overall + per bucket)
  - adaptive: per-horizon trailing-window inverse-MSE weights from RECENTLY OBSERVED
    origins (o <= D-3-h) — legal (past obs only), deterministic.

Dry-run window = winter 2024-25 (the scored regime). Emits official pred_matrix.csv +
mse_summary.csv for the chosen rule, plus the contest metric MSE_1-5d / MSE_6-10d.

OUT: outputs/metrics/18-framework.md/.json ; outputs/submission_dryrun/{pred_matrix,mse_summary}.csv
"""
from __future__ import annotations
import json, sys, time, traceback, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import mse
from src.data import load_wide_daily
from src.models import lightgbm_train, lightgbm_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "work" / "data"
OUT_M = ROOT / "work" / "outputs" / "metrics"; OUT_M.mkdir(parents=True, exist_ok=True)
OUT_S = ROOT / "work" / "outputs" / "submission_dryrun"; OUT_S.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 11)); REPORTING_LAG = 3
DRYRUN = list(range(626, 716))     # winter 2024-25 origins (90)
ET_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
LOG = []
def log(m):
    s = f"[{time.strftime('%H:%M:%S')}] {m}"; print(s, flush=True); LOG.append(s)


def main():
    t0 = time.time()
    wide = load_wide_daily(); d2i = {d: i for i, d in enumerate(wide["midday_day"].to_list())}
    y_full = wide["estimated_avoidable_deaths"].to_numpy(); total = wide.shape[0]

    # store preds: (D, h) -> (y_true, p_et, p_lgb)
    P = {}
    for h in HORIZONS:
        fm = pl.read_parquet(DATA / f"feature_matrix_h{h}.parquet")
        idx = np.array([d2i[d] for d in fm["decision_day"].to_list()])
        full = [c for c in fm.columns if c not in ("decision_day", "label_day", "y")]
        endo = [c for c in full if " " not in c and "(" not in c]
        th = time.time(); nfit = 0
        for D in DRYRUN:
            try:
                tr = idx <= (D - REPORTING_LAG - h)      # observable training labels only
                te = idx == D
                if tr.sum() < 30 or te.sum() == 0:
                    continue
                Xtr_f = fm.filter(pl.Series(tr)).select(full); ytr = fm.filter(pl.Series(tr))["y"].to_numpy()
                Xte_f = fm.filter(pl.Series(te)).select(full)
                Xtr_e = fm.filter(pl.Series(tr)).select(endo).to_numpy(); Xte_e = fm.filter(pl.Series(te)).select(endo).to_numpy()
                yte = float(fm.filter(pl.Series(te))["y"].to_numpy()[0])
                mdl, cm = lightgbm_train(Xtr_f, ytr); p_lgb = float(lightgbm_predict(mdl, Xte_f, mapping=cm)[0])
                imp = SimpleImputer(strategy="median"); a = imp.fit_transform(Xtr_e); b = imp.transform(Xte_e)
                # NOTE: log1p(y) on ExtraTrees helped in fold-validation (script 19) but did NOT
                # transfer to this strict per-origin deployment setup (it slightly hurt the blend),
                # so we keep the raw target. Lesson: validate in the deployment setup, not just CV.
                et = ExtraTreesRegressor(**ET_PARAMS); et.fit(a, ytr); p_et = float(et.predict(b)[0])
                P[(D, h)] = (yte, p_et, p_lgb); nfit += 1
            except Exception:
                log(f"  !! origin {D} h={h} FAILED:\n{traceback.format_exc()}")
        log(f"  h={h}: {nfit} origins fit ({time.time()-th:.0f}s)")
        _dump(P)   # incremental

    # ---------- combination rules (post-hoc, no refit) ----------
    origins = [D for D in DRYRUN if all((D, h) in P for h in HORIZONS)]
    log(f"scorable origins: {len(origins)}")

    def bucket_mse(predfn):
        # predfn(D,h)->yhat ; returns (mse_1_5, mse_6_10) over all origins
        e15, e610 = [], []
        for D in origins:
            for h in HORIZONS:
                yt = P[(D, h)][0]; yh = predfn(D, h)
                (e15 if h <= 5 else e610).append((yt - yh) ** 2)
        return float(np.mean(e15)), float(np.mean(e610))

    rules = {}
    rules["pure_ET"] = bucket_mse(lambda D, h: P[(D, h)][1])
    rules["pure_LGB"] = bucket_mse(lambda D, h: P[(D, h)][2])
    for w in (0.3, 0.5, 0.6, 0.7, 0.8):
        rules[f"fixed_ET{w:.1f}"] = bucket_mse(lambda D, h, w=w: w * P[(D, h)][1] + (1 - w) * P[(D, h)][2])

    # adaptive: per-h trailing-window (W) inverse-MSE on observed origins (o <= D-3-h)
    W = 15
    def adaptive_pred(D, h):
        hist = [o for o in origins if o <= D - REPORTING_LAG - h and o < D]
        hist = hist[-W:]
        if len(hist) < 5:
            return 0.6 * P[(D, h)][1] + 0.4 * P[(D, h)][2]
        e_et = np.mean([(P[(o, h)][0] - P[(o, h)][1]) ** 2 for o in hist])
        e_lgb = np.mean([(P[(o, h)][0] - P[(o, h)][2]) ** 2 for o in hist])
        ie, il = 1.0 / (e_et + 1e-9), 1.0 / (e_lgb + 1e-9)
        w = ie / (ie + il)
        return w * P[(D, h)][1] + (1 - w) * P[(D, h)][2]
    rules["adaptive"] = bucket_mse(adaptive_pred)

    # per-bucket best fixed (in-sample on dry-run; flagged)
    log("rule comparison (dry-run winter):")
    for r, (m15, m610) in rules.items():
        log(f"  {r:14s} 1-5d={m15:.4f} 6-10d={m610:.4f} overall={(m15+m610)/2:.4f}")

    # FROZEN combination (team decision 2026-06-05): per-horizon "weight A" —
    # 0.55 ExtraTrees for days 1-5 (LightGBM's metrics help the short horizons),
    # 0.9 ExtraTrees for days 6-10 (metrics go stale; ExtraTrees carries the signal).
    def weightA_pred(D, h):
        w = 0.55 if h <= 5 else 0.9
        return w * P[(D, h)][1] + (1 - w) * P[(D, h)][2]
    rules["weightA"] = bucket_mse(weightA_pred)
    chosen = "weightA"
    log(f"CHOSEN rule: {chosen}  -> 1-5d={rules[chosen][0]:.4f} 6-10d={rules[chosen][1]:.4f}")

    # emit official format for chosen rule
    predfn = {"pure_ET": lambda D, h: P[(D, h)][1],
              "adaptive": adaptive_pred,
              "weightA": weightA_pred,
              "fixed_ET0.6": lambda D, h: 0.6 * P[(D, h)][1] + 0.4 * P[(D, h)][2],
              "fixed_ET0.7": lambda D, h: 0.7 * P[(D, h)][1] + 0.3 * P[(D, h)][2]}[chosen]
    pm, ms = [], []
    for fid, D in enumerate(origins, 1):
        row = {"forecast_id": fid}
        preds = [predfn(D, h) for h in HORIZONS]
        for h in HORIZONS: row[f"day_{h}"] = preds[h - 1]
        pm.append(row)
        a = np.array([P[(D, h)][0] for h in HORIZONS]); p = np.array(preds)
        ms.append({"forecast_id": fid, "mse_1_5": float(np.mean((a[:5] - p[:5]) ** 2)),
                   "mse_6_10": float(np.mean((a[5:] - p[5:]) ** 2))})
    pl.DataFrame(pm).write_csv(OUT_S / "pred_matrix.csv")
    pl.DataFrame(ms).write_csv(OUT_S / "mse_summary.csv")

    out = {"rules": rules, "chosen": chosen, "n_origins": len(origins), "W": W, "log": LOG}
    (OUT_M / "18-framework.json").write_text(json.dumps(out, indent=2, default=str))
    md = ["# Deployment framework — winter dry-run (official format)\n",
          f"_Re-fit ExtraTrees_cal + LightGBM_full per origin (expanding, y to D-3). "
          f"{len(origins)} winter origins. Runtime {time.time()-t0:.0f}s._\n",
          "## Combination rules (contest metric on winter dry-run)\n",
          "| rule | MSE_1-5d | MSE_6-10d | overall |", "|---|--:|--:|--:|"]
    for r, (m15, m610) in sorted(rules.items(), key=lambda x: x[1][0] + x[1][1]):
        md.append(f"| {r} | {m15:.4f} | {m610:.4f} | {(m15+m610)/2:.4f} |")
    md += [f"\n**Chosen: `{chosen}`** → MSE_1-5d {rules[chosen][0]:.4f}, MSE_6-10d {rules[chosen][1]:.4f}.\n",
           "_Fixed-weight rules with a per-bucket-optimal w are in-sample on the dry-run; "
           "`adaptive` and `pure_ET` are the deployment-honest choices._\n",
           f"Official-format `pred_matrix.csv` + `mse_summary.csv` written to outputs/submission_dryrun/ "
           f"({len(origins)} forecast origins).\n",
           "_vs sNaive winter floor (stage 1): 1-5d 0.2438, 6-10d 0.3096._"]
    (OUT_M / "18-framework.md").write_text("\n".join(md))
    log(f"DONE chosen={chosen} ({time.time()-t0:.0f}s)")


def _dump(P):
    try:
        (OUT_M / "18-framework-preds.json").write_text(
            json.dumps({f"{D}|{h}": v for (D, h), v in P.items()}, default=str))
    except Exception:
        pass


if __name__ == "__main__":
    main()
