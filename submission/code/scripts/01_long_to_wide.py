"""01 — Long → wide → daily, then save to Parquet.

Entry point for Phase 1.1.  Mirrors the official R baseline's preprocessing,
so any later step that reads `work/data/wide_daily.parquet` knows what's in
there. Run with `./work/run.sh python work/scripts/01_long_to_wide.py`.
"""

### Takes raw data in long format, converts to wide daily format, and saves as Parquet file.
### i.e. each row is one day

from __future__ import annotations ### readability 
import json ### work with json data (text)
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import long_to_wide_daily, WIDE_DAILY, DATA_RAW  # noqa: E402

### Define where project root folder is
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "work" / "runs"

def main() -> None:
    ### Create new folder for run
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M") + "-long-to-wide"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ### Convert long to wide, and time how long it takes
    t0 = time.time()
    wide = long_to_wide_daily()
    duration = time.time() - t0

    # Basic acceptance checks (PLAN.md §1.1)
    ### Check n.o. rows and columns
    n_rows, n_cols = wide.shape
    ### ensure outcome column is in the data
    has_outcome = "estimated_avoidable_deaths" in wide.columns
    ### check for missing values
    n_outcome_missing = wide["estimated_avoidable_deaths"].null_count() if has_outcome else None

    ### print summary of results
    print(f"=== 01 long → wide ===")
    print(f"  rows × cols : {n_rows} × {n_cols}")
    print(f"  first date  : {wide['midday_day'].min()}")
    print(f"  last  date  : {wide['midday_day'].max()}")
    print(f"  has Y col   : {has_outcome}")
    print(f"  Y nulls     : {n_outcome_missing}")
    print(f"  wrote       : {WIDE_DAILY}")
    print(f"  duration    : {duration:.1f}s")

    ### saves info as JSON file
    run_log = {
        "run_id": run_id,
        "input_csv": str(DATA_RAW),
        "output_parquet": str(WIDE_DAILY),
        "shape_rows": n_rows,
        "shape_cols": n_cols,
        "first_date": str(wide["midday_day"].min()),
        "last_date": str(wide["midday_day"].max()),
        "outcome_column": "estimated_avoidable_deaths",
        "outcome_nulls": n_outcome_missing,
        "duration_seconds": duration,
    }
    (run_dir / "run.json").write_text(json.dumps(run_log, indent=2))
    print(f"  log         : {run_dir / 'run.json'}")


if __name__ == "__main__":
    main()
