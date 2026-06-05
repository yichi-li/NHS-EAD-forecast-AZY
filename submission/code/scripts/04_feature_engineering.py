"""04 — Build per-horizon feature matrices for h ∈ 1..10.

Output: 10 files at work/data/feature_matrix_h{h}.parquet, each leakage-safe
(see tests/test_no_leakage.py).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_wide_daily  # noqa: E402
from src.features import build_features_for_horizon  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "work" / "data"
OUT_FEATURES = PROJECT_ROOT / "work" / "outputs" / "features"
RUNS_DIR = PROJECT_ROOT / "work" / "runs"
OUT_FEATURES.mkdir(parents=True, exist_ok=True)

### predict 1,2,3,4,5,6,7,8,9,10 days ahead
### 10 seperate feature matrices
### different features per horizon prevents leakage
HORIZONS = list(range(1, 11))


def main():
    ### load data and catalog
    t0 = time.time()
    wide = load_wide_daily()
    catalog = pl.read_parquet(OUT_FEATURES / "feature_catalog.parquet")
    lag0_path = PROJECT_ROOT / "work" / "outputs" / "eda" / "05-lag-correlations.csv"

    print(f"Loaded wide {wide.shape}, catalog {catalog.shape}")
    print(f"Building per-horizon matrices for h ∈ {HORIZONS}")

    ### build feature matrix for each horizon, and save
    all_meta = {}
    for h in HORIZONS:
        ts = time.time()
        fm, meta = build_features_for_horizon(
            wide,
            catalog,
            h=h,
            top_n_x_for_rolling=50,
            lag0_pearson_path=lag0_path,
        )
        out_path = DATA_DIR / f"feature_matrix_h{h}.parquet"
        fm.write_parquet(out_path, compression="zstd")
        print(f"  h={h:2d}: {fm.shape[0]:4d} rows × {fm.shape[1]:4d} cols → {out_path.name}  ({time.time()-ts:.1f}s)")
        all_meta[str(h)] = meta

    duration = time.time() - t0
    print(f"\nTotal time: {duration:.1f}s")

    # Save combined meta
    meta_path = OUT_FEATURES / "feature_matrix_meta.json"
    meta_path.write_text(json.dumps({
        "horizons": HORIZONS,
        "per_horizon": all_meta,
        "duration_seconds": duration,
        "generated": datetime.now().isoformat(),
    }, indent=2, default=str))
    print(f"Meta:  {meta_path}")

    # Run log
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M") + "-fe-perH"
    (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / run_id / "run.json").write_text(json.dumps({
        "horizons": HORIZONS,
        "rows_per_h": {str(h): all_meta[str(h)]["n_rows"] for h in HORIZONS},
        "cols_per_h": {str(h): all_meta[str(h)]["n_cols"] for h in HORIZONS},
        "duration_seconds": duration,
    }, indent=2))


if __name__ == "__main__":
    main()
