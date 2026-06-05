"""Rolling-origin CV with explicit holdout, per M5 Finding 6.

The split scheme (matching the contest's structure):

  Full dev set (930 days)
  ┌──────────────────────────────────────────────────┬─────────────┐
  │  Training/CV pool (870 days)                     │  HOLDOUT    │
  │  ┌─ Rolling-origin folds (K × M test days) ─┐   │  (60 days)  │
  │  │                                          │   │  untouched  │
  └──┴──────────────────────────────────────────┴───┴─────────────┘

- The competition will release a 182-day assessment set on 2026-06-06; that
  is the ultimate held-out test set.
- Internally we reserve the last 60 days of the dev set as an *honest* holdout
  that NO model selection / HPO is allowed to touch. It is opened ONCE in
  Phase 3.3.x for the final v2 evaluation number.
- The rest is used for rolling-origin CV during development.

Within the CV pool, K folds each have a 10-day test window at the end:

  Fold 0: train [0, T_0); test decision days [T_0, T_0+M)
  Fold 1: train [0, T_1); test decision days [T_1, T_1+M)
  ...
  with T_k = (CV_end) - (K - k) * M

For per-horizon matrices, "train rows" for h-matrix are rows whose label_day
falls BEFORE the test cutoff — i.e., decision_day < T_k - h.

Per M5 Finding 6, we track BOTH mean AND std of MSE across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


HOLDOUT_DAYS = 60          # last N days of dev set, untouched until 3.3.x
DEFAULT_K_FOLDS = 4         # number of CV folds
DEFAULT_TEST_M = 10         # test decision days per fold (mimics one 10-day forecast window)


@dataclass
class HoldoutSplit:
    """Split a 930-day dev period into a CV pool and an honest holdout."""

    n_holdout: int = HOLDOUT_DAYS

    def split(self, total_days: int) -> tuple[slice, slice]:
        """Return (cv_slice, holdout_slice) over decision_day indices."""
        if total_days <= self.n_holdout:
            raise ValueError(f"total_days {total_days} too small for {self.n_holdout}-day holdout")
        return slice(0, total_days - self.n_holdout), slice(total_days - self.n_holdout, total_days)


@dataclass
class Fold:
    fold_idx: int
    train_end_day: int          # row-index of cutoff (exclusive), in the CV pool
    test_decision_days: list[int]  # row-indices of decision days to predict


@dataclass
class RollingOriginCV:
    """K non-overlapping rolling-origin folds at the END of the CV pool."""

    k: int = DEFAULT_K_FOLDS
    test_m: int = DEFAULT_TEST_M

    def folds(self, n_cv_days: int) -> list[Fold]:
        if n_cv_days < self.k * self.test_m + 100:
            raise ValueError(
                f"CV pool too small ({n_cv_days}) for k={self.k}, test_m={self.test_m}"
            )
        out: list[Fold] = []
        for k_idx in range(self.k):
            # Fold 0 is oldest, fold k-1 is most recent
            test_start = n_cv_days - (self.k - k_idx) * self.test_m
            test_end = test_start + self.test_m
            train_end = test_start
            out.append(Fold(
                fold_idx=k_idx,
                train_end_day=train_end,
                test_decision_days=list(range(test_start, test_end)),
            ))
        return out


def train_row_mask_for_fold(
    decision_day_indices: np.ndarray,
    fold: Fold,
    h: int,
) -> np.ndarray:
    """For h-matrix, training rows are those with label_day < fold.train_end_day.

    Equivalently: decision_day_idx + h < train_end_day, i.e., decision_day_idx < train_end_day - h.

    Returns a boolean mask aligned with decision_day_indices.
    """
    return decision_day_indices < (fold.train_end_day - h)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


@dataclass
class FoldResult:
    fold_idx: int
    h: int
    n_test: int
    mse_h: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)


def summarise_per_horizon(results: list[FoldResult]) -> dict:
    """Aggregate per-fold per-horizon MSE into:
       - mse_per_h: dict h → mean ± std across folds
       - mse_1_5d_mean / std, mse_6_10d_mean / std (averaged across (h, fold) pairs)
       - mse_overall_mean / std
    """
    by_h: dict[int, list[float]] = {}
    for r in results:
        by_h.setdefault(r.h, []).append(r.mse_h)

    mse_per_h = {
        h: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
        for h, v in by_h.items()
    }

    h_in_15 = [h for h in by_h if 1 <= h <= 5]
    h_in_610 = [h for h in by_h if 6 <= h <= 10]
    flat_15 = [v for h in h_in_15 for v in by_h[h]]
    flat_610 = [v for h in h_in_610 for v in by_h[h]]
    flat_all = [v for h in by_h for v in by_h[h]]

    return {
        "mse_per_h": mse_per_h,
        "mse_1_5d_mean": float(np.mean(flat_15)) if flat_15 else float("nan"),
        "mse_1_5d_std": float(np.std(flat_15)) if flat_15 else float("nan"),
        "mse_6_10d_mean": float(np.mean(flat_610)) if flat_610 else float("nan"),
        "mse_6_10d_std": float(np.std(flat_610)) if flat_610 else float("nan"),
        "mse_overall_mean": float(np.mean(flat_all)) if flat_all else float("nan"),
        "mse_overall_std": float(np.std(flat_all)) if flat_all else float("nan"),
    }
