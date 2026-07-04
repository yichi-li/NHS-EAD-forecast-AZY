"""Model wrappers — baselines (sNaive, ES_bu) and LightGBM.

Each function returns predictions for the test window given a train window.
Predictions are made for horizons 1..H (default H=10), one prediction per
day in the test window.

For LightGBM we use the YJ_STU-style param block, with `min_data_in_leaf`
scaled down for our smaller dataset.
"""

from __future__ import annotations

from typing import Sequence

import lightgbm as lgb
import numpy as np
import polars as pl


# YJ_STU's hyperparameters (`docs/literature/m5-winner-1-yj-stu.md` §4 Passage 5),
# with min_data_in_leaf reduced ~10× because our dataset is ~930 days vs M5's 60M rows.
# Subsample/feature_fraction kept aggressive for overfitting control.
LGB_PARAMS_BASELINE = {
    "boosting_type": "gbdt",
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "subsample": 0.5,
    "subsample_freq": 1,
    "learning_rate": 0.015,
    "num_leaves": 255,           # ↓ from 2047 (smaller dataset)
    "min_data_in_leaf": 100,     # ↓ from 4095
    "feature_fraction": 0.5,
    "max_bin": 100,
    "n_estimators": 800,         # ↓ from 3000 (we have less data → fewer trees needed)
    "boost_from_average": False,
    "verbose": -1,
    "force_col_wise": True,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# sNaive — seasonal-naive at weekly cycle
#
# Fixed 2026-05-26 (was buggy in v2.0):
#   - Off-by-one in the DOW lookup: predicted D+1 used y(D-7) instead of y(D-6).
#   - Did not respect the 3-day reporting lag: included y(D-1), y(D-2) in lookup.
#
# New API: caller passes the OBSERVABLE y series (already excluding the
# reporting-lag tail) and a start_offset describing where the prediction begins
# relative to the last observable day.
# ---------------------------------------------------------------------------
def snaive_forecast(
    y_observable: np.ndarray,
    horizon: int = 10,
    season: int = 7,
    start_offset: int = 1,
) -> np.ndarray:
    """Seasonal-naive forecast.

    Let `t` be the index of the last observable day (y_observable[-1] == y(t)).
    Returns predictions for y(t + start_offset), y(t + start_offset + 1), ...,
    y(t + start_offset + horizon - 1).

    Standard formula:  y_hat(t + k) = y(t + k - season * ceil(k / season))

    Parameters
    ----------
    y_observable : np.ndarray
        The OBSERVABLE y series (already excludes the reporting-lag tail).
    horizon : int
        Number of forecast steps to return.
    season : int
        Seasonal period. 7 for weekly.
    start_offset : int
        First forecast step relative to the last observable day. Defaults to 1
        (predict the next day onwards). For the SPHERE-PPL contest, where the
        target has a 3-day reporting lag (last observable is y(D-3)) and we
        predict D+1..D+10, call with start_offset=4, horizon=10.
    """
    import math

    if len(y_observable) < season:
        return np.full(
            horizon,
            float(np.nanmean(y_observable)) if len(y_observable) else 0.0,
        )

    preds = np.zeros(horizon)
    n = len(y_observable)
    for i, offset in enumerate(range(start_offset, start_offset + horizon)):
        # y_hat(t + offset) = y(t + offset - season * k) where k = ceil(offset / season)
        k = math.ceil(offset / season)
        position = (n - 1) + offset - season * k
        if 0 <= position < n:
            preds[i] = float(y_observable[position])
        else:
            preds[i] = float(np.nanmean(y_observable))
    return preds


# ---------------------------------------------------------------------------
# ES (exponential smoothing, weekly seasonality) — uses statsforecast
#
# Fixed 2026-05-26 (was buggy in v2.0): did not respect 3-day reporting lag.
# ---------------------------------------------------------------------------
def es_forecast(
    y_observable: np.ndarray,
    horizon: int = 10,
    start_offset: int = 1,
    season: int = 7,
) -> np.ndarray:
    """Holt-Winters / AutoETS forecast at weekly seasonality.

    statsforecast.AutoETS handles automatic model selection (additive/multiplicative,
    with/without trend/season). We force season_length=7 for weekly NHS pattern.

    See `snaive_forecast` for the y_observable / start_offset / horizon semantics.
    """
    from statsforecast.models import AutoETS

    y = np.asarray(y_observable, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 14:
        return np.full(horizon, float(y[-1]) if len(y) else 0.0)

    model = AutoETS(season_length=season)
    model.fit(y)
    # Predict enough steps to cover [start_offset .. start_offset + horizon - 1]
    total_h = start_offset + horizon - 1
    fc = model.predict(h=total_h)
    full_preds = np.asarray(fc["mean"], dtype=float)
    return full_preds[start_offset - 1 : start_offset - 1 + horizon]


# ---------------------------------------------------------------------------
# LightGBM — uses the engineered feature matrix
# ---------------------------------------------------------------------------
def _sanitize_col(c: str) -> str:
    """LightGBM rejects special chars in column names; replace them with underscores."""
    import re
    out = re.sub(r"[^A-Za-z0-9_]", "_", c)
    # Collapse runs of _
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "col"


def _sanitize_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, str]]:
    """Rename columns to LightGBM-safe names. Returns the renamed df and the mapping."""
    mapping = {}
    used = set()
    for c in df.columns:
        new = _sanitize_col(c)
        # Avoid collision: append a suffix if needed
        base = new
        i = 1
        while new in used:
            new = f"{base}_{i}"
            i += 1
        used.add(new)
        mapping[c] = new
    return df.rename(mapping), mapping


def lightgbm_train(
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    params: dict | None = None,
    early_stopping_rounds: int | None = None,
    X_valid: pl.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
) -> tuple[lgb.Booster, dict[str, str]]:
    """Train a LightGBM model. Returns (booster, col_rename_map).

    Sanitises column names so they pass LightGBM's special-char check.
    No early stopping by default (per YJ_STU's recipe).
    """
    params = params or LGB_PARAMS_BASELINE
    X_train_s, mapping = _sanitize_columns(X_train)
    train_data = lgb.Dataset(X_train_s.to_pandas(), label=y_train)

    callbacks = []
    valid_sets = None
    if X_valid is not None and y_valid is not None:
        X_valid_s = X_valid.rename(mapping)
        valid_sets = [lgb.Dataset(X_valid_s.to_pandas(), label=y_valid)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=params.get("n_estimators", 1000),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    return model, mapping


def lightgbm_predict(model: lgb.Booster, X: pl.DataFrame, mapping: dict[str, str] | None = None) -> np.ndarray:
    if mapping is not None:
        X = X.rename(mapping)
    pred = model.predict(X.to_pandas())
    # Clip to non-negative (since target is a non-negative count) — Tweedie usually gives ≥0 but be safe
    return np.maximum(pred, 0.0)
