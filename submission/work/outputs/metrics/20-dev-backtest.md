# Development backtest (most recent 173 dev origins)

The frozen framework (weight A: 0.55 ExtraTrees + 0.45 LightGBM for days 1-5,
0.90 ExtraTrees + 0.10 LightGBM for days 6-10) evaluated on the most recent 173
development origins (2025-04-01 .. 2025-09-20; all ten target days fall within the
development window <= 30 Sep 2025), the same rolling-origin basis other entrants
report for a development backtest.

| horizon block | MSE   |
|---------------|-------|
| 1-5d          | 0.040 |
| 6-10d         | 0.046 |

Exact values: MSE_1-5d = 0.0401, MSE_6-10d = 0.0464 (173 origins).
Reproduce: from `work/`, run `python dev_backtest.py` after 01 / gen_lag0 / 03 / 04.
