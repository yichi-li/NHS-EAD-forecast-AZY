"""Feature engineering pipeline — per-horizon, leakage-safe (v2).

Builds one feature matrix per forecast horizon h ∈ {1..10}. Each row of the
resulting `feature_matrix_h{h}.parquet` represents a *decision day D*. The
features are everything observable at time D-midday; the label is y(D+h).

Leakage rules (verified by tests/test_no_leakage.py):

- **Y reporting lag = 3 days.** At decision day D, the most recent observable
  y is y(D-3). Therefore all y-derived features must be expressible from
  y(D-3), y(D-4), ..., not y(D-2), y(D-1), y(D).
- **X observed up to D-midday.** At decision day D, x(D) is observable
  (midday cut implied by `data.py`). Therefore x_lag_0 = x(D), x_lag_1 = x(D-1),
  ..., are all safe. Future x is NOT observable.
- **Calendar of label day is known in advance.** Bank holidays, day-of-week,
  flu season for D+h are public knowledge at any earlier time, so
  calendar(D+h) is a legal feature.

This module is the single source of truth for feature engineering. Any later
script should call `build_features_for_horizon(wide, catalog, h=...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import holidays
import numpy as np
import polars as pl


# Y-side: reporting lag is 3 days. Earliest observable y at decision day D is y(D-3).
Y_LAG_BASE = 3   # minimum y-lag (D-3 = most recent observable y at decision time D)
Y_LAG_MAX = 17   # we keep 15 lags: y_lag_3, y_lag_4, ..., y_lag_17
Y_ROLLING_WINDOWS = [7, 14, 30, 60]

# X-side: observable up to D-midday. No additional shift needed.
X_LAGS = [0, 1, 7]
X_ROLLING_WINDOWS = [7, 14, 30]


def _calendar_features(label_dates: pl.Series, fourier: bool = False) -> dict[str, np.ndarray]:
    """Compute calendar features for the LABEL day D+h (known in advance).

    If fourier=True, also add smooth weekly + annual seasonality via sin/cos
    Fourier terms. These give the tree a continuous periodic signal instead of
    only the discrete dow/month splits — the calendar of D+h is public knowledge
    at decision time D, so this is leakage-safe.
    """
    s = label_dates.to_pandas()
    yrs = range(int(s.dt.year.min()), int(s.dt.year.max()) + 1)
    uk = holidays.country_holidays("GB", years=yrs)
    feats = {
        "dow": s.dt.dayofweek.to_numpy().astype(np.int8),
        "dom": s.dt.day.to_numpy().astype(np.int8),
        "wom": (((s.dt.day - 1) // 7) + 1).to_numpy().astype(np.int8),
        "month": s.dt.month.to_numpy().astype(np.int8),
        "year": (s.dt.year - s.dt.year.min()).to_numpy().astype(np.int8),
        "is_weekend": (s.dt.dayofweek >= 5).astype(np.int8).to_numpy(),
        "is_uk_bank_holiday": np.array([1 if d in uk else 0 for d in s], dtype=np.int8),
        "day_of_year": s.dt.dayofyear.to_numpy().astype(np.int16),
        "is_christmas_period": (
            ((s.dt.month == 12) & (s.dt.day >= 24)) | ((s.dt.month == 1) & (s.dt.day <= 2))
        ).astype(np.int8).to_numpy(),
        "is_flu_season": ((s.dt.month <= 3) | (s.dt.month >= 10)).astype(np.int8).to_numpy(),
    }
    if fourier:
        dow = s.dt.dayofweek.to_numpy().astype(float)        # 0..6
        doy = s.dt.dayofyear.to_numpy().astype(float)        # 1..365/366
        for k in (1, 2, 3):                                  # 3 weekly harmonics
            feats[f"fourier_week_sin{k}"] = np.sin(2 * np.pi * k * dow / 7.0)
            feats[f"fourier_week_cos{k}"] = np.cos(2 * np.pi * k * dow / 7.0)
        for k in (1, 2):                                     # 2 annual harmonics
            feats[f"fourier_year_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            feats[f"fourier_year_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return feats


def build_features_for_horizon(
    wide: pl.DataFrame,
    catalog: pl.DataFrame,
    *,
    h: int,
    top_n_x_for_rolling: int = 50,
    lag0_pearson_path: Path | None = None,
    fourier: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Build feature matrix for direct forecasting at horizon h.

    Parameters
    ----------
    wide : pl.DataFrame
        Output of `data.long_to_wide_daily()`. One row per midday_day, columns
        are metric-by-coverage strings plus the column `estimated_avoidable_deaths`.
    catalog : pl.DataFrame
        Output of `feature_catalog.build_catalog(wide.columns)`.
    h : int
        Forecast horizon in days (1..10).
    top_n_x_for_rolling : int
        Number of top leading-indicator x metrics to apply rolling stats to.
    lag0_pearson_path : Path | None
        Optional path to EDA's `05-lag-correlations.csv`. If provided, used to
        rank x metrics by |lag-0 Pearson| for selective rolling.

    Returns
    -------
    fm : pl.DataFrame
        One row per decision day D (where y(D+h) exists in `wide`).
        Columns: decision_day, label_day, y (= y(D+h)), and the engineered features.
    meta : dict
        Family counts and provenance for the FEATURES report.
    """
    if h < 1 or h > 30:
        raise ValueError(f"horizon h must be in 1..30, got {h}")

    # Sort by date
    wide = wide.sort("midday_day").clone()
    n_rows = wide.shape[0]

    # ------------------------------------------------------------------
    # Build features at each decision day D, then shift to align label = y(D+h)
    # ------------------------------------------------------------------

    # Start fresh — base table has the decision-day metadata only
    df = wide.select(["midday_day"]).rename({"midday_day": "decision_day"}).clone()
    df = df.with_columns(
        (pl.col("decision_day") + pl.duration(days=h)).alias("label_day"),
    )

    # ---- 1. Target features (y_lag_n for n = Y_LAG_BASE..Y_LAG_MAX, y_rolling) ----
    # At decision day D (index i), y_lag_n = wide['y'][i - n], so we just shift.
    y_full = wide["estimated_avoidable_deaths"]

    for n in range(Y_LAG_BASE, Y_LAG_MAX + 1):
        df = df.with_columns(y_full.shift(n).alias(f"y_lag_{n}"))

    for w in Y_ROLLING_WINDOWS:
        # rolling_mean computes the mean of the window ending at the current row.
        # We want the rolling mean ending at D - Y_LAG_BASE, so shift by Y_LAG_BASE.
        df = df.with_columns(
            y_full.rolling_mean(window_size=w).shift(Y_LAG_BASE).alias(
                f"y_rmean_{w}_shift{Y_LAG_BASE}"
            ),
            y_full.rolling_std(window_size=w).shift(Y_LAG_BASE).alias(
                f"y_rstd_{w}_shift{Y_LAG_BASE}"
            ),
        )

    # ---- 2. X features ----
    x_cols = [c for c in wide.columns if c not in ("midday_day", "estimated_avoidable_deaths")]

    # 2a. Lag features for all x columns
    lag_exprs = []
    for c in x_cols:
        for lag_n in X_LAGS:
            # x_lag_n at row i = x[i - lag_n], observable at decision day D (= row i)
            lag_exprs.append(wide[c].shift(lag_n).alias(f"{c}_lag_{lag_n}"))
    df = df.with_columns(lag_exprs)

    # 2b. Rolling stats, only on top-N leading indicators
    if lag0_pearson_path and lag0_pearson_path.exists():
        lag_df = pl.read_csv(lag0_pearson_path).filter(pl.col("lag") == 0)
        lag_df = lag_df.with_columns(pl.col("pearson").abs().alias("abs_pearson"))
        top_x = lag_df.sort("abs_pearson", descending=True).head(top_n_x_for_rolling)["metric"].to_list()
    else:
        top_x = x_cols[:top_n_x_for_rolling]

    roll_exprs = []
    for c in top_x:
        if c not in wide.columns:
            continue
        for w in X_ROLLING_WINDOWS:
            roll_exprs.append(wide[c].rolling_mean(window_size=w).alias(f"{c}_rmean_{w}"))
            roll_exprs.append(wide[c].rolling_std(window_size=w).alias(f"{c}_rstd_{w}"))
    df = df.with_columns(roll_exprs)

    # ---- 3. Causal-group aggregates (mean and std of z-scored metrics, observable at D) ----
    # Computed from x at D (no shift). z-score uses dev-period mean/std (legal:
    # these are constants known once dev period is fixed).
    catalog_x = catalog.filter(pl.col("causal_role") != "target")
    for role in ["upstream", "concurrent", "downstream"]:
        role_cols = catalog_x.filter(pl.col("causal_role") == role)["column"].to_list()
        role_cols = [c for c in role_cols if c in wide.columns]
        if not role_cols:
            continue

        # Standardise each col by its dev-period mean/std, then take per-row
        # mean/std across the role's columns. Dev-period statistics are
        # constants known at deployment, not future-leaking.
        z_arrays = []
        for c in role_cols:
            col = wide[c].to_numpy().astype(float)
            mean = np.nanmean(col)
            std = np.nanstd(col)
            if std > 0 and not np.isnan(std):
                z_arrays.append((col - mean) / std)
            else:
                z_arrays.append(np.zeros_like(col))
        zvals = np.column_stack(z_arrays)
        with np.errstate(invalid="ignore"):
            df = df.with_columns(
                pl.Series(f"{role}_pressure_mean", np.nanmean(zvals, axis=1)),
                pl.Series(f"{role}_pressure_std", np.nanstd(zvals, axis=1)),
            )

    # ---- 4. Calendar features for the LABEL day (D+h) ----
    cal = _calendar_features(df["label_day"], fourier=fourier)
    for k, v in cal.items():
        df = df.with_columns(pl.Series(k, v))

    # ---- 5. The label — y(D+h) — comes from shifting wide's y BACKWARD by h ----
    # y at row i (= decision day D_i) for label = y[i + h]
    df = df.with_columns(y_full.shift(-h).alias("y"))

    # ---- 6. Drop rows where the label is null (i.e. last h rows of dev period) ----
    df = df.filter(pl.col("y").is_not_null())

    # ---- 7. Reorder: decision_day, label_day, y, then features ----
    other = [c for c in df.columns if c not in ("decision_day", "label_day", "y")]
    df = df.select(["decision_day", "label_day", "y"] + other)

    # ---- 8. Meta ----
    feature_cols = [c for c in df.columns if c not in ("decision_day", "label_day", "y")]
    meta = {
        "horizon": h,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "n_features": len(feature_cols),
        "y_lag_range": [Y_LAG_BASE, Y_LAG_MAX],
        "y_rolling_windows": Y_ROLLING_WINDOWS,
        "y_rolling_shift": Y_LAG_BASE,
        "x_lags": X_LAGS,
        "x_rolling_windows": X_ROLLING_WINDOWS,
        "n_x_metrics": len(x_cols),
        "top_n_x_for_rolling": top_n_x_for_rolling,
        "feature_families": {
            "y_lag": Y_LAG_MAX - Y_LAG_BASE + 1,
            "y_rolling": 2 * len(Y_ROLLING_WINDOWS),
            "x_lag": len(x_cols) * len(X_LAGS),
            "x_rolling": 2 * len(top_x) * len(X_ROLLING_WINDOWS),
            "causal_aggregate": 6,
            "calendar": 10,
        },
    }
    return df, meta


# -----------------------------------------------------------------------------
# Legacy wrapper — kept so any unmigrated caller still works (deprecated)
# -----------------------------------------------------------------------------
def build_features(wide, catalog, *, h: int = 1, **kw):
    """DEPRECATED — use build_features_for_horizon. This wrapper is kept for
    backward compat during the v1→v2 transition and will be removed in v2.x."""
    import warnings
    warnings.warn(
        "build_features() is deprecated. Use build_features_for_horizon(h=...) "
        "and load the matrix that matches your forecast horizon.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_features_for_horizon(wide, catalog, h=h, **kw)
