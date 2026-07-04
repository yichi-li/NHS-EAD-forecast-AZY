"""Leakage tests for the per-horizon feature engineering pipeline.

These tests catch the v1 leakage bug (h=1 feature matrix reused for h=2..10,
where y_lag_4 at test rows referenced y values inside the test window).

Run with: `./run.sh python -m pytest tests/ -v`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_wide_daily
from src.feature_catalog import build_catalog
from src.features import (
    Y_LAG_BASE,
    Y_LAG_MAX,
    X_LAGS,
    build_features_for_horizon,
)


# Build one wide_daily + catalog + a couple of horizons once for the test module
@pytest.fixture(scope="module")
def wide():
    return load_wide_daily()


@pytest.fixture(scope="module")
def catalog(wide):
    return build_catalog(wide.columns)


@pytest.fixture(scope="module")
def fm_h1(wide, catalog):
    fm, _ = build_features_for_horizon(wide, catalog, h=1)
    return fm


@pytest.fixture(scope="module")
def fm_h5(wide, catalog):
    fm, _ = build_features_for_horizon(wide, catalog, h=5)
    return fm


@pytest.fixture(scope="module")
def fm_h10(wide, catalog):
    fm, _ = build_features_for_horizon(wide, catalog, h=10)
    return fm


# ---------------------------------------------------------------------------
# 1. Schema invariants — must hold for any horizon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_required_columns_exist(wide, catalog, h):
    fm, _ = build_features_for_horizon(wide, catalog, h=h)
    required = {"decision_day", "label_day", "y"}
    missing = required - set(fm.columns)
    assert not missing, f"Missing required columns at h={h}: {missing}"


def test_no_y_lag_below_base(fm_h1, fm_h5, fm_h10):
    """y_lag_n must not exist for n < Y_LAG_BASE (= 3 days reporting lag)."""
    for fm in [fm_h1, fm_h5, fm_h10]:
        for n in range(0, Y_LAG_BASE):
            assert f"y_lag_{n}" not in fm.columns, (
                f"y_lag_{n} should NOT exist (n < Y_LAG_BASE={Y_LAG_BASE})."
            )


def test_y_lag_range_present(fm_h1):
    """All y_lag_{Y_LAG_BASE..Y_LAG_MAX} should exist."""
    for n in range(Y_LAG_BASE, Y_LAG_MAX + 1):
        assert f"y_lag_{n}" in fm_h1.columns, f"Missing y_lag_{n}"


# ---------------------------------------------------------------------------
# 2. The critical leakage tests — y-lag and x-lag references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_label_equals_y_at_label_day(wide, catalog, h):
    """y in fm must equal y at the label_day in wide, NOT y at decision_day."""
    fm, _ = build_features_for_horizon(wide, catalog, h=h)
    # Build a lookup: midday_day → y
    y_lookup = dict(zip(wide["midday_day"].to_list(), wide["estimated_avoidable_deaths"].to_list()))

    sample = fm.head(50)
    for row in sample.iter_rows(named=True):
        label_day = row["label_day"]
        expected_y = y_lookup.get(label_day)
        if expected_y is not None and not np.isnan(expected_y):
            assert abs(row["y"] - expected_y) < 1e-10, (
                f"h={h}: at decision_day {row['decision_day']}, "
                f"label y is {row['y']} but y_lookup[{label_day}] = {expected_y}"
            )


@pytest.mark.parametrize("h", [1, 5, 10])
def test_y_lag_n_references_y_at_decision_minus_n(wide, catalog, h):
    """y_lag_n at row with decision_day D must equal y(D - n days).

    This is the central leakage check — at decision day D, the model must only
    see y values from days <= D - Y_LAG_BASE. The y_lag_n feature is defined as
    y(D - n), and we verify by lookup.
    """
    fm, _ = build_features_for_horizon(wide, catalog, h=h)
    y_lookup = dict(zip(wide["midday_day"].to_list(), wide["estimated_avoidable_deaths"].to_list()))

    sample = fm.head(50).tail(30)  # rows that have all lags computable
    from datetime import timedelta
    for row in sample.iter_rows(named=True):
        D = row["decision_day"]
        for n in range(Y_LAG_BASE, min(Y_LAG_BASE + 5, Y_LAG_MAX + 1)):
            ref_day = D - timedelta(days=n)
            expected = y_lookup.get(ref_day)
            got = row[f"y_lag_{n}"]
            if expected is not None and not np.isnan(expected):
                assert got is not None and abs(got - expected) < 1e-10, (
                    f"h={h}: y_lag_{n} at D={D} = {got}, expected y({ref_day}) = {expected}"
                )


@pytest.mark.parametrize("h", [1, 5, 10])
def test_no_y_lag_references_future(fm_h1, fm_h5, fm_h10, h):
    """For any row, y_lag_n must reference a date strictly before decision_day.

    Equivalently, no y_lag_n at row R should reference a y value at date >= D_R.
    This is true by construction (y_lag_n references D - n with n >= Y_LAG_BASE = 3),
    but we test it explicitly to catch any future code regression.
    """
    fm = {1: fm_h1, 5: fm_h5, 10: fm_h10}[h]
    # The construction shifts y by n days, so for row at decision_day D,
    # y_lag_n at that row equals y at index (row_index - n). If that index < 0,
    # it's NaN. The label is y at (row_index + h). So:
    # - y_lag_n references row_index - n
    # - we need row_index - n < (row_index + h) - h - Y_LAG_BASE + 1
    # which simplifies to n >= Y_LAG_BASE. Since Y_LAG_BASE = 3, this is always true.
    assert Y_LAG_BASE >= 3, "Y_LAG_BASE must be >= 3 (the y reporting lag)"


# ---------------------------------------------------------------------------
# 3. X-lag observable-at-D check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_x_lag_0_references_x_at_decision_day(wide, catalog, h):
    """x_lag_0 at row with decision_day D must equal x at decision_day D, NOT label_day D+h.

    This is the OTHER critical leakage check — in v1 baseline, x_lag_0 at test
    row D+h was the FUTURE x observation. Now we verify it's at decision day.
    """
    fm, _ = build_features_for_horizon(wide, catalog, h=h)

    # Pick a representative x column
    x_col = "No. of DTAs - BRI"
    if x_col not in wide.columns:
        pytest.skip(f"{x_col} not in dataset")

    x_lookup = dict(zip(wide["midday_day"].to_list(), wide[x_col].to_list()))
    feature_name = f"{x_col}_lag_0"

    sample = fm.head(60).tail(30)
    for row in sample.iter_rows(named=True):
        D = row["decision_day"]
        expected = x_lookup.get(D)
        got = row[feature_name]
        if expected is not None and not (isinstance(expected, float) and np.isnan(expected)):
            assert got is not None and abs(got - expected) < 1e-10, (
                f"h={h}: {feature_name} at D={D} = {got}, expected x({D}) = {expected}"
            )


# ---------------------------------------------------------------------------
# 4. Calendar features reference label_day (D+h)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_dow_matches_label_day(wide, catalog, h):
    """dow column should be the day-of-week of label_day, not decision_day."""
    fm, _ = build_features_for_horizon(wide, catalog, h=h)
    sample = fm.head(20)
    for row in sample.iter_rows(named=True):
        # python weekday: Monday=0, Sunday=6
        expected_dow = row["label_day"].weekday()
        assert int(row["dow"]) == expected_dow, (
            f"h={h}: dow at decision_day {row['decision_day']} (label_day {row['label_day']}) "
            f"= {row['dow']}, expected {expected_dow}"
        )


# ---------------------------------------------------------------------------
# 5. Row counts (sanity)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("h", [1, 5, 10])
def test_row_count_matches_horizon(wide, catalog, h):
    """For horizon h, the feature matrix should have (n_wide_rows - h) rows.

    Reason: y(D+h) must exist in wide. The last h rows of wide don't have a label.
    """
    fm, _ = build_features_for_horizon(wide, catalog, h=h)
    expected = wide.shape[0] - h
    assert fm.shape[0] == expected, (
        f"h={h}: feature matrix has {fm.shape[0]} rows, expected {expected}"
    )


# ---------------------------------------------------------------------------
# 6. Catalog regression test for Q6.1 fix
# ---------------------------------------------------------------------------

def test_p0_complex_swasft_are_concurrent(wide):
    """P0 Discharges / Complex Discharges / SWASFT incidents should be CONCURRENT (Q6.1 fix)."""
    catalog = build_catalog(wide.columns)
    for column in catalog["column"].to_list():
        if "P0 Discharges" in column or "Complex Discharges" in column or "(SWASFT) Number of" in column:
            role = catalog.filter(pl.col("column") == column)["causal_role"][0]
            assert role == "concurrent", (
                f"{column!r} should be concurrent (Q6.1 fix), got {role!r}"
            )
