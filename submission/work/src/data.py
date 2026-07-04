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
DATA_VALIDATION = PROJECT_ROOT / "NHS-EAD-forecast-main" / "data" / "turingAI_forecasting_challenge_validation_dataset.csv"
DATA_PROCESSED = PROJECT_ROOT / "work" / "data"
WIDE_DAILY = DATA_PROCESSED / "wide_daily.parquet"

# Dev / assessment cut per competition rules (README.md §"Data > Assessment dataset"):
# Dev:        2023-03-16 to 2025-09-30
# Assessment: 2025-10-01 to 2026-02-17 in the released validation set (real values).
#             The organisers' amended validation dataset ends 17 Feb 2026, giving
#             131 sliding 10-day assessment periods. In the development-only release
#             this same period is filled with dummy -9999.
DEV_END = "2025-09-30"
ASSESSMENT_DUMMY = -9999


def _scan_long(csv: Path) -> pl.LazyFrame:
    """Lazy-scan an official long-format CSV and parse `dt` to a tz-naive datetime.

    Handles both source formats with one code path:
    - development CSV : "YYYY-MM-DD HH:MM:SS" and date-only "YYYY-MM-DD"
    - validation CSV : "YYYY-MM-DDTHH:MM:SSZ" (ISO 8601, UTC)
    We normalise the ISO 'T'/'Z' (T -> space, Z -> "") so a single format string
    parses both, keeping every timestamp tz-naive so the two frames concatenate.
    """
    lf = pl.scan_csv(
        csv,
        try_parse_dates=False,
        null_values=["NA", ""],
        schema_overrides={"value": pl.Float64},
    )
    lf = lf.with_columns(
        pl.col("dt").str.replace("T", " ", literal=True).str.replace("Z", "", literal=True).alias("dt")
    )
    # The R baseline uses parse_date_time(orders = c("Ymd HMS", "Ymd")): try the
    # full datetime first, fall back to date-only for the remainder.
    return lf.with_columns(
        pl.coalesce(
            pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
            pl.col("dt").str.to_datetime("%Y-%m-%d", strict=False),
        ).alias("dt"),
    )


def long_to_wide_daily(
    src_csv: Path = DATA_RAW,
    dst_parquet: Path = WIDE_DAILY,
    *,
    dev_only: bool = True,
    validation_csv: Path | None = DATA_VALIDATION,
) -> pl.DataFrame:
    """Build the wide-daily Parquet from the official long-format CSV(s).

    Parameters
    ----------
    src_csv : Path
        The official development long-format CSV (1.8 GB).
    dst_parquet : Path
        Output Parquet path; parent dir is created.
    dev_only : bool
        If True, drop rows with `dt > DEV_END` (the assessment-period rows are
        dummy -9999 in this release anyway, but trimming early saves memory).
    validation_csv : Path | None
        The released validation/assessment CSV (1 Oct 2025 .. 17 Feb 2026, real
        values). When present it is appended to the (dev-trimmed) development data,
        so the wide table spans the full timeline needed to forecast the 131
        assessment periods. Defaults to DATA_VALIDATION; ignored if the file is
        absent (then the table is development-only, unchanged behaviour).

    Returns
    -------
    pl.DataFrame
        The wide-daily DataFrame that was written to disk.
    """
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)

    # 1-2. Lazy-scan + parse dt (both source date formats handled in _scan_long).
    df = _scan_long(src_csv)

    # Trim the dev file to the development window. Its assessment-period rows are
    # dummy -9999 anyway; when the released validation data is appended below we
    # must not let the dummy dev rows overlap the real assessment rows.
    use_validation = validation_csv is not None and Path(validation_csv).exists()
    if dev_only or use_validation:
        df = df.filter(pl.col("dt") <= pl.lit(f"{DEV_END} 23:59:59").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    # Append the released validation/assessment data so the wide table reaches
    # 17 Feb 2026 (the amended end point -> 131 sliding 10-day periods).
    if use_validation:
        df = pl.concat([df, _scan_long(Path(validation_csv))], how="vertical_relaxed")

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

    # 7b. Deterministic column order. The pivot's column order follows the
    # (streaming) scan order and therefore varies run-to-run; LightGBM's split
    # tie-breaking is sensitive to feature order, so an unfixed order makes
    # forecasts differ slightly between reproductions. Sorting the columns once
    # here makes the whole pipeline reproducible. Downstream code selects columns
    # by name, so this reordering is safe.
    wide = wide.select(["midday_day"] + sorted(c for c in wide.columns if c != "midday_day"))

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
