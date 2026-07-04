"""gen_lag0.py — build the dev-only lag-0 correlation file used to select which
metrics receive rolling features (scripts/04 -> src/features.py top_x).

Leakage-safe: the correlations are computed on the DEVELOPMENT window only
(midday_day <= DEV_END), so the assessment period never influences feature
selection. Mirrors scripts/02_eda.py::lag_correlations, restricted to the dev
window. Output: outputs/eda/05-lag-correlations.csv.

Run after scripts/01 and before scripts/04.
"""
from __future__ import annotations
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
from src.data import load_wide_daily, DEV_END  # noqa: E402

TARGET = "estimated_avoidable_deaths"
LAGS = [0, 1, 3, 7, 14, 28]
OUT = WORK / "outputs" / "eda" / "05-lag-correlations.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

wide = load_wide_daily()
dev_end = dt.date(*(int(x) for x in DEV_END.split("-")))
dev = wide.filter(pl.col("midday_day") <= dev_end)
print(f"dev portion: {dev.shape}  ({dev['midday_day'].min()} .. {dev['midday_day'].max()})", flush=True)

df_pd = dev.to_pandas().set_index("midday_day").sort_index()
metrics = [c for c in df_pd.columns if c != TARGET]
y = df_pd[TARGET]

rows = []
for m in metrics:
    x = df_pd[m]
    for lag in LAGS:
        x_lagged = x.shift(lag)
        both = pd.DataFrame({"y": y, "x": x_lagged}).dropna()
        if len(both) < 50:
            continue
        with np.errstate(invalid="ignore"):
            pear = float(np.corrcoef(both["y"].to_numpy(), both["x"].to_numpy())[0, 1])
            spear = float(stats.spearmanr(both["y"].to_numpy(), both["x"].to_numpy()).statistic)
        rows.append({"metric": m, "lag": lag, "pearson": pear, "spearman": spear, "n": len(both)})

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"wrote {OUT}  rows={len(rows)}  lag0_rows={sum(1 for r in rows if r['lag'] == 0)}", flush=True)
