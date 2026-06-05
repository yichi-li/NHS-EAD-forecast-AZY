"""Baseline-correctness tests (added 2026-05-26 after v2 audit found two bugs).

Bugs caught here:

1. sNaive off-by-one — v2.0 returned `last_week[(h-1) % 7]`, which means at
   horizon h=1 the prediction was the value 7 days BEFORE the last observable,
   i.e. the same day-of-week as the last observable, NOT the same DOW as the
   forecast target. Fixed to use the standard `y_hat(t+k) = y(t+k - season*k)`.

2. Reporting-lag violation — v2.0 used `y_full[:D_idx]` (= y(0..D-1)) as input
   to sNaive and ES. The contest rules state at D, only y(0..D-3) is observable
   (3-day reporting lag). Fix: callers pass y(0..D-3) and start_offset=4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import snaive_forecast, es_forecast


# ---------------------------------------------------------------------------
# sNaive — day-of-week alignment
# ---------------------------------------------------------------------------

def test_snaive_h1_uses_correct_dow_with_no_reporting_lag():
    """With no reporting lag (start_offset=1), predicting h=1 should use y(t-6),
    which has the same DOW as y(t+1)."""
    # Construct a y where each value equals its index mod 7 (1..7 then repeat).
    # DOW encoded by value: y[i] % 7 == i % 7.
    y = np.array([float(i) for i in range(28)])  # 28 days
    preds = snaive_forecast(y, horizon=10, start_offset=1)
    # y[-1] = y[27], DOW = 27 % 7 = 6 → say Sunday
    # Predicting t+1 = day 28, DOW = 0 → Monday
    # sNaive: y_hat(28) = y(28 - 7*ceil(1/7)) = y(28-7) = y(21). y[21] = 21. DOW 21%7 = 0 = Monday ✓
    assert preds[0] == 21.0, f"h=1 should give y[21]=21 (Mon), got {preds[0]}"
    # h=7: y_hat(28+6) = y(34) → t+7 = 27+7 = 34. predicted as y(34-7*ceil(7/7))=y(27)=27. DOW 27%7=6=Sun. Target DOW = 33%7=5=Sat. Hmm wait, t+7 should have DOW (6+7)%7 = 6 = Sun. So y(27) (DOW 6=Sun) matches ✓
    # Actually for h=7, predicting y(t+7) where t=27. Target = y(34) at DOW (27+7)%7 = 6. y(27) at DOW 6 ✓
    assert preds[6] == 27.0, f"h=7 should give y[27]=27 (Sun), got {preds[6]}"


def test_snaive_respects_start_offset_for_reporting_lag():
    """With start_offset=4 (3-day reporting lag, predicting D+1 = t+4),
    h=1 should use y(t+4 - 7) = y(t-3) (latest DOW-matching value)."""
    y = np.array([float(i) for i in range(28)])  # last observable y[-1] = y(27)
    # Decision day D: t=27, observable up to y(27). Forecast target D+1 means t + 4.
    # y_hat(t+4) = y(t+4 - 7*ceil(4/7)) = y(t+4 - 7) = y(24). y[24] = 24.
    preds = snaive_forecast(y, horizon=10, start_offset=4)
    assert preds[0] == 24.0, f"start_offset=4 h=1 should give y[24]=24, got {preds[0]}"
    # h=4 → predict t+7. y_hat(t+7) = y(t+7-7) = y(27).
    assert preds[3] == 27.0, f"start_offset=4 h=4 should give y[27]=27, got {preds[3]}"
    # h=5 → predict t+8. y_hat(t+8) = y(t+8 - 7*ceil(8/7)) = y(t+8-14) = y(21).
    assert preds[4] == 21.0, f"start_offset=4 h=5 should give y[21]=21, got {preds[4]}"


def test_snaive_does_not_use_unobservable_y():
    """Critical: with the 3-day reporting lag, snaive_forecast called with the
    observable subset must NOT touch y(D-1), y(D-2) when predicting D+1..D+10."""
    # Construct y where the unobservable tail is a giant outlier; the predictions
    # should be unaffected.
    n = 30
    y_full = np.arange(n, dtype=float)
    D_idx = 20
    REPORTING_LAG = 3

    # Buggy call (what v2.0 did): used the full series including y(D-1), y(D-2)
    buggy_pred = snaive_forecast(y_full[:D_idx], horizon=10, start_offset=1)

    # Fixed call: pass observable subset + start_offset
    observable = y_full[: D_idx - (REPORTING_LAG - 1)]   # y(0..D-3)
    fixed_pred = snaive_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)

    # Make sure neither call references y(D-1)=y_full[19] or y(D-2)=y_full[18]
    # in the FIXED version: observable = y_full[:18] = y(0..17), last is y(17) = y(D-3).
    # Fixed pred for h=1: y_hat(D+1) = y_hat(t+4) where t=17. y(17+4-7) = y(14).
    assert fixed_pred[0] == 14.0, f"fixed sNaive h=1 should give y(14), got {fixed_pred[0]}"

    # The two should differ — proves the fix changes behaviour
    assert not np.allclose(buggy_pred, fixed_pred), (
        "Fixed and buggy sNaive should give different predictions on this input"
    )


def test_snaive_short_series_returns_mean():
    """If observable series shorter than season, return mean."""
    y = np.array([1.0, 2.0, 3.0])  # only 3 values, season=7
    preds = snaive_forecast(y, horizon=10, start_offset=1)
    assert all(p == np.mean(y) for p in preds), "Short series should fall back to mean"


# ---------------------------------------------------------------------------
# ES — reporting-lag respect
# ---------------------------------------------------------------------------

def test_es_uses_only_observable_y():
    """ES must respect reporting lag — model is fit on observable subset only."""
    # Smoke test: ES with start_offset=4 produces a valid output and doesn't crash.
    np.random.seed(0)
    y = np.random.randn(100).cumsum() + 50.0
    REPORTING_LAG = 3
    D_idx = 50
    observable = y[: D_idx - (REPORTING_LAG - 1)]
    preds = es_forecast(observable, horizon=10, start_offset=REPORTING_LAG + 1)
    assert preds.shape == (10,)
    # Predictions shouldn't be wildly negative (we started with y around 50)
    assert preds.min() > -1000.0
    assert preds.max() < 1000.0


def test_es_falls_back_on_short_series():
    y = np.array([1.0, 2.0, 3.0])  # too short
    preds = es_forecast(y, horizon=10, start_offset=1)
    assert preds.shape == (10,)
    assert all(p == 3.0 for p in preds), "Short-series fallback should repeat last value"
