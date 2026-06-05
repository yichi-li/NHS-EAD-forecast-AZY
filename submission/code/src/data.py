"""Data loading and long → wide → daily aggregation.

Mirrors the official R baseline (`NHS_example_forecast.R` lines 28–35):
- Parse `dt` to datetime UTC.
- Define `midday_day` = same date if hour <= 12, else date + 1.
- Group by (midday_day, metric_name, coverage_label), take mean.
- Pivot to wide: one row per midday_day, one column per metric_name × coverage_label.

This module is the single source of truth for the long-to-wide step. Any later
script that wants the wide daily data should call `load_wide_daily()`.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "NHS-EAD-forecast-main" / "data" / "turingAI_forecasting_challenge_dataset.csv"
DATA_PROCESSED = PROJECT_ROOT / "work" / "data"
WIDE_DAILY = DATA_PROCESSED / "wide_daily.parquet"

# Dev / assessment cut per competition rules (README.md §"Data > Assessment dataset"):
# Dev:        2023-03-16 to 2025-09-30
# Assessment: 2025-10-01 to 2026-03-31 (dummy -9999 in the development release)
DEV_END = "2025-09-30"
ASSESSMENT_DUMMY = -9999


def long_to_wide_daily(
    src_csv: Path = DATA_RAW,
    dst_parquet: Path = WIDE_DAILY,
    *,
    dev_only: bool = True,
) -> pl.DataFrame:
    """Build the wide-daily Parquet from the official long-format CSV.

    Parameters
    ----------
    src_csv : Path
        The official long-format CSV (1.8 GB).
    dst_parquet : Path
        Output Parquet path; parent dir is created.
    dev_only : bool
        If True, drop rows with `dt > DEV_END` (the assessment-period rows are
        dummy -9999 in this release anyway, but trimming early saves memory).

    Returns
    -------
    pl.DataFrame
        The wide-daily DataFrame that was written to disk.
    """
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)

    # 1. Lazy scan with NA handling (the file contains literal "NA" strings)
    df = pl.scan_csv(
        src_csv,
        try_parse_dates=False,
        null_values=["NA", ""],
        schema_overrides={"value": pl.Float64},
    )

    # 2. Parse dt to datetime — file has two formats: "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DD".
    # The R baseline uses parse_date_time(orders = c("Ymd HMS", "Ymd")).
    # Strategy: try full datetime first, fall back to date-only for the remainder.
    df = df.with_columns(
        pl.coalesce(
            pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("dt").str.to_datetime("%Y-%m-%d", strict=False),
        ).alias("dt"),
    )
    if dev_only:
        df = df.filter(pl.col("dt") <= pl.lit(f"{DEV_END} 23:59:59").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    # midday_day = date(dt) + (1 day if hour > 12)
    df = df.with_columns(
        pl.when(pl.col("dt").dt.hour() <= 12)
        .then(pl.col("dt").dt.date())
        .otherwise(pl.col("dt").dt.date() + pl.duration(days=1))
        .alias("midday_day"),
    )

    # 3. Drop dummy outcome rows just in case (some metric coverages could have -9999)
    df = df.filter(pl.col("value") != ASSESSMENT_DUMMY)

    # 4. Construct a single column name = metric_name + " - " + coverage_label
    #    (Identical to how `metric_metadata.csv` builds metric_label)
    df = df.with_columns(
        (pl.col("metric_name") + pl.lit(" - ") + pl.col("coverage_label")).alias("col"),
    )

    # 5. Aggregate to daily means
    daily = (
        df.group_by(["midday_day", "col"])
        .agg(pl.col("value").mean().alias("value"))
    )

    # 6. Pivot to wide. Collect lazily first so polars can plan.
    daily_df = daily.collect(engine="streaming")
    wide = daily_df.pivot(
        index="midday_day",
        on="col",
        values="value",
        aggregate_function="first",  # rows are already grouped, this is a no-op
    ).sort("midday_day")

    # 7. Rename the outcome column to a canonical short name
    outcome_candidates = [c for c in wide.columns if c.lower().startswith("estimated_avoidable_deaths")]
    if outcome_candidates:
        wide = wide.rename({outcome_candidates[0]: "estimated_avoidable_deaths"})

    # 8. Persist
    wide.write_parquet(dst_parquet, compression="zstd")
    return wide


def load_wide_daily(path: Path = WIDE_DAILY) -> pl.DataFrame:
    """Load the wide-daily Parquet built by `long_to_wide_daily()`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_long_to_wide.py first."
        )
    return pl.read_parquet(path)


if __name__ == "__main__":
    import time
    t0 = time.time()
    wide = long_to_wide_daily()
    dt = time.time() - t0
    print(f"Wrote {WIDE_DAILY}")
    print(f"  shape: {wide.shape}")
    print(f"  columns: {len(wide.columns)}")
    print(f"  time: {dt:.1f}s")
