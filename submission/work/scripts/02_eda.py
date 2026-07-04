"""02 — EDA on wide_daily.parquet.

Produces:
  outputs/eda/01-missingness.png      — heatmap
  outputs/eda/02-y-distribution.png    — Y histogram + density
  outputs/eda/03-y-timeseries.png      — Y over time
  outputs/eda/04-time-coverage.png     — per-metric first/last non-null date
  outputs/eda/05-lag-correlations.csv  — top correlations across lags {0,1,3,7,14,28}
  outputs/eda/EDA-REPORT.md            — narrative summary with stats + decisions
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_wide_daily  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "work" / "outputs" / "eda"
RUNS_DIR = PROJECT_ROOT / "work" / "runs"
EDA_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "estimated_avoidable_deaths"
LAGS = [0, 1, 3, 7, 14, 28]

### create missingness heatmap
def plot_missingness(df: pl.DataFrame) -> dict:
    """Per-metric × time missingness heatmap, plus per-metric missing rate summary."""
    df_pd = df.to_pandas().set_index("midday_day")
    is_null = df_pd.isna().astype(int)

    # Order columns by missingness for readability
    miss_pct = is_null.mean().sort_values()
    is_null = is_null[miss_pct.index]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        is_null.T,
        cmap="Greys",
        cbar_kws={"label": "1 = missing"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(f"Missingness: {len(is_null.columns)} metrics × {len(is_null)} days\n(rows sorted by overall missing-rate; darker = missing)")
    ax.set_xlabel("Day (2023-03-16 → 2025-09-30)")
    ax.set_ylabel("Metric (×coverage)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01-missingness.png", dpi=120)
    plt.close(fig)

    return {
        "n_metrics": int(len(miss_pct)),
        "n_fully_covered": int((miss_pct == 0).sum()),
        "n_partial": int(((miss_pct > 0) & (miss_pct < 1)).sum()),
        "n_empty": int((miss_pct == 1).sum()),
        "mean_missing_pct": float(miss_pct.mean()),
        "median_missing_pct": float(miss_pct.median()),
        "top10_most_missing": miss_pct.tail(10).to_dict(),
        "top10_best_covered": miss_pct.head(10).to_dict(),
    }

### histogram for how skewed y is
def plot_y_distribution(df: pl.DataFrame) -> dict:
    """Y histogram + density + skewness/kurtosis. Drives Tweedie vs L2 choice."""
    y = df[TARGET].to_numpy()
    y_valid = y[~np.isnan(y)]

    skew = float(stats.skew(y_valid))
    kurt = float(stats.kurtosis(y_valid))
    fraction_zero = float((y_valid == 0).mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(y_valid, bins=40, edgecolor="black", alpha=0.7)
    axes[0].axvline(np.mean(y_valid), color="red", linestyle="--", label=f"mean = {np.mean(y_valid):.3f}")
    axes[0].axvline(np.median(y_valid), color="orange", linestyle="--", label=f"median = {np.median(y_valid):.3f}")
    axes[0].set_title(f"Y histogram\nn={len(y_valid)}, skew={skew:.2f}, kurtosis={kurt:.2f}, zero-fraction={fraction_zero:.2%}")
    axes[0].set_xlabel("estimated_avoidable_deaths (daily)")
    axes[0].set_ylabel("count")
    axes[0].legend()

    sns.kdeplot(y_valid, ax=axes[1], fill=True)
    axes[1].set_title("Y density")
    axes[1].set_xlabel("estimated_avoidable_deaths (daily)")

    fig.tight_layout()
    fig.savefig(EDA_DIR / "02-y-distribution.png", dpi=120)
    plt.close(fig)

    return {
        "n_valid": int(len(y_valid)),
        "mean": float(np.mean(y_valid)),
        "median": float(np.median(y_valid)),
        "min": float(np.min(y_valid)),
        "max": float(np.max(y_valid)),
        "std": float(np.std(y_valid)),
        "skewness": skew,
        "kurtosis": kurt,
        "fraction_zero": fraction_zero,
        "p25": float(np.quantile(y_valid, 0.25)),
        "p75": float(np.quantile(y_valid, 0.75)),
        "p95": float(np.quantile(y_valid, 0.95)),
    }


### plot avoidable deaths over time
def plot_y_timeseries(df: pl.DataFrame) -> dict:
    """Y over time — show trend, seasonality, regime shifts."""
    dates = df["midday_day"].to_pandas()
    y = df[TARGET].to_numpy()

    # weekday means
    df_pd = df.select(["midday_day", TARGET]).to_pandas()
    df_pd["dow"] = df_pd["midday_day"].astype("datetime64[ns]").dt.day_name()
    dow_means = df_pd.groupby("dow")[TARGET].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].plot(dates, y, linewidth=0.5)
    # Rolling mean overlay
    rolling = (
        df.with_columns(pl.col(TARGET).rolling_mean(window_size=14).alias("rmean14"))
        ["rmean14"].to_numpy()
    )
    axes[0].plot(dates, rolling, color="red", linewidth=2, label="14-day rolling mean")
    axes[0].set_title("estimated_avoidable_deaths over time (dev period)")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("avoidable deaths / day")
    axes[0].legend()

    dow_means.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
    axes[1].set_title("Mean Y by day-of-week")
    axes[1].set_ylabel("mean Y")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(EDA_DIR / "03-y-timeseries.png", dpi=120)
    plt.close(fig)

    return {
        "weekday_means": {k: float(v) for k, v in dow_means.to_dict().items()},
        "max_y_date": str(df_pd.loc[df_pd[TARGET].idxmax(), "midday_day"]),
        "max_y_value": float(df_pd[TARGET].max()),
        "min_y_date": str(df_pd.loc[df_pd[TARGET].idxmin(), "midday_day"]),
        "min_y_value": float(df_pd[TARGET].min()),
    }


### check when each variable starts and ends
def plot_time_coverage(df: pl.DataFrame) -> dict:
    """Per-metric first/last non-null date. Identifies metrics that started/ended mid-period."""
    df_pd = df.to_pandas().set_index("midday_day")
    metrics = [c for c in df_pd.columns if c != TARGET]

    coverage = []
    for m in metrics:
        non_null = df_pd[m].dropna().index
        if len(non_null) > 0:
            coverage.append({"metric": m, "first": non_null.min(), "last": non_null.max(), "n_obs": len(non_null)})
        else:
            coverage.append({"metric": m, "first": None, "last": None, "n_obs": 0})

    import pandas as pd
    cov_df = pd.DataFrame(coverage)
    cov_df = cov_df.sort_values("first", na_position="last").reset_index(drop=True)

    # Plot: each metric a horizontal bar from first to last
    fig, ax = plt.subplots(figsize=(14, 12))
    for i, row in cov_df.iterrows():
        if row["first"] is not None:
            ax.plot([row["first"], row["last"]], [i, i], color="steelblue", linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel("date")
    ax.set_title(f"Per-metric time coverage ({len(metrics)} metrics, sorted by first observation)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "04-time-coverage.png", dpi=120)
    plt.close(fig)

    full = cov_df[cov_df["n_obs"] >= 0.95 * 930]
    return {
        "n_metrics": len(metrics),
        "n_full_coverage_95pct": int(len(full)),
        "n_started_late": int((cov_df["first"] > df_pd.index.min()).sum()),
        "n_ended_early": int((cov_df["last"] < df_pd.index.max()).sum()),
    }

### check lag correlations for each metric (with Y, at lags {0,1,3,7,14,28})
def lag_correlations(df: pl.DataFrame) -> tuple[dict, list[dict]]:
    """For each metric, compute Pearson + Spearman correlation with Y at each lag.

    Restricted to the DEVELOPMENT window (midday_day <= 2025-09-30): this file also
    feeds the rolling-feature selection (05-lag-correlations.csv), which must be
    leakage-safe, so the assessment period must not influence it. Same rule as
    gen_lag0.py.
    """
    import datetime as _dt
    df = df.filter(pl.col("midday_day") <= _dt.date(2025, 9, 30))
    df_pd = df.to_pandas().set_index("midday_day").sort_index()
    metrics = [c for c in df_pd.columns if c != TARGET]
    y = df_pd[TARGET]

    rows = []
    for m in metrics:
        x = df_pd[m]
        for lag in LAGS:
            # Positive lag = predictor leading; y at time t, x at time t-lag
            x_lagged = x.shift(lag)
            both = pl.from_pandas(
                __import__("pandas").DataFrame({"y": y, "x": x_lagged}).dropna()
            )
            if len(both) < 50:
                continue
            with np.errstate(invalid="ignore"):
                pearson = float(np.corrcoef(both["y"].to_numpy(), both["x"].to_numpy())[0, 1])
                spearman = float(
                    stats.spearmanr(both["y"].to_numpy(), both["x"].to_numpy()).statistic
                )
            rows.append({"metric": m, "lag": lag, "pearson": pearson, "spearman": spearman, "n": len(both)})

    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(EDA_DIR / "05-lag-correlations.csv", index=False)

    ### Top 30 by absolute pearson, at any lag
    lag_df["abs_pearson"] = lag_df["pearson"].abs()
    top = lag_df.sort_values("abs_pearson", ascending=False).head(30)
    top_list = top[["metric", "lag", "pearson", "spearman", "n"]].to_dict(orient="records")

    summary = {
        "n_metric_lag_pairs": len(lag_df),
        "top30_by_abs_pearson_at_any_lag": top_list,
        "lags_tested": LAGS,
    }
    return summary, top_list

### plotting seasonality patterns for estimated avoidable deaths
def plot_seasonality(df: pl.DataFrame) -> dict:
    """Monthly and weekly seasonality patterns for estimated avoidable deaths."""
    df_pd = df.select(["midday_day", TARGET]).to_pandas()
    df_pd["midday_day"] = df_pd["midday_day"].astype("datetime64[ns]")
    df_pd["month"] = df_pd["midday_day"].dt.month
    df_pd["month_name"] = df_pd["midday_day"].dt.strftime("%b")
    df_pd["year"] = df_pd["midday_day"].dt.year
    df_pd["week"] = df_pd["midday_day"].dt.isocalendar().week.astype(int)

    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = df_pd.groupby("month_name")[TARGET].agg(["mean","std"]).reindex(month_order)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ### Monthly averages with error bars
    axes[0].bar(month_order, monthly["mean"], yerr=monthly["std"],
                color="steelblue", edgecolor="black", alpha=0.8, capsize=4)
    axes[0].set_title("Mean estimated avoidable deaths by month (± 1 std)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Mean avoidable deaths per day")
    axes[0].tick_params(axis="x", rotation=45)

    ### Year-over-year overlay
    for year, grp in df_pd.groupby("year"):
        monthly_yr = grp.groupby("month")[TARGET].mean()
        axes[1].plot(monthly_yr.index, monthly_yr.values, marker="o", label=str(year))
    axes[1].set_title("Year-over-year monthly pattern")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Mean avoidable deaths per day")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_order, rotation=45)
    axes[1].legend()

    ### Weekly pattern across the year
    weekly = df_pd.groupby("week")[TARGET].mean()
    axes[2].plot(weekly.index, weekly.values, linewidth=1.5, color="steelblue")
    axes[2].set_title("Mean estimated avoidable deaths by week of year")
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("Mean avoidable deaths per day")

    fig.tight_layout()
    fig.savefig(EDA_DIR / "06-seasonality.png", dpi=120)
    plt.close(fig)

    winter_months = ["Dec", "Jan", "Feb"]
    summer_months = ["Jun", "Jul", "Aug"]
    winter_mean = monthly.loc[winter_months, "mean"].mean()
    summer_mean = monthly.loc[summer_months, "mean"].mean()

    return {
        "monthly_means": {k: float(v) for k, v in monthly["mean"].to_dict().items()},
        "peak_month": monthly["mean"].idxmax(),
        "trough_month": monthly["mean"].idxmin(),
        "winter_mean": float(winter_mean),
        "summer_mean": float(summer_mean),
        "winter_summer_ratio": float(winter_mean / summer_mean) if summer_mean > 0 else None,
    }

### plotting autocorrelation patterns for estimated avoidable deaths
### (understand how much history matters for forecasting)
def plot_autocorrelation(df: pl.DataFrame) -> dict:
    """ACF and PACF plots for estimated avoidable deaths — shows how far back in time matters for forecasting."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf

    y = df.sort("midday_day")[TARGET].to_numpy()
    y_valid = y[~np.isnan(y)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    plot_acf(y_valid, lags=60, ax=axes[0], alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF) — how much does estimated avoidable deaths at lag k predict today's estimated avoidable deaths?")
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Correlation")

    plot_pacf(y_valid, lags=60, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF) — direct effect at each lag, controlling for shorter lags")
    axes[1].set_xlabel("Lag (days)")
    axes[1].set_ylabel("Partial Correlation")

    fig.tight_layout()
    fig.savefig(EDA_DIR / "07-autocorrelation.png", dpi=120)
    plt.close(fig)

    # Compute significant lags
    acf_vals = acf(y_valid, nlags=60, fft=True)
    pacf_vals = pacf(y_valid, nlags=60, method="ywm")
    confidence_bound = 1.96 / np.sqrt(len(y_valid))

    significant_acf_lags = [i for i, v in enumerate(acf_vals) if abs(v) > confidence_bound and i > 0]
    significant_pacf_lags = [i for i, v in enumerate(pacf_vals) if abs(v) > confidence_bound and i > 0]

    return {
        "significant_acf_lags": significant_acf_lags[:10],
        "significant_pacf_lags": significant_pacf_lags[:10],
        "acf_lag1": float(acf_vals[1]),
        "acf_lag7": float(acf_vals[7]),
        "acf_lag14": float(acf_vals[14]),
        "acf_lag28": float(acf_vals[28]),
        "confidence_bound": float(confidence_bound),
    }

### plotting outliers for estimated avoidable deaths
### (identify days with unusually high or low estimated avoidable deaths)
### Z score = measured how far a value is from average
### IQR = inter quartile range
def plot_outliers(df: pl.DataFrame) -> dict:
    """Detect and visualise outliers in estimated avoidable deaths using IQR and Z-score methods."""

    df_pd = df.select(["midday_day", TARGET]).to_pandas()
    df_pd["midday_day"] = df_pd["midday_day"].astype("datetime64[ns]")
    y = df_pd[TARGET]

    ### IQR method
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    iqr_lower = Q1 - 3 * IQR
    iqr_upper = Q3 + 3 * IQR
    iqr_outliers = df_pd[(y < iqr_lower) | (y > iqr_upper)].copy()

    ### Z-score method
    z_scores = (y - y.mean()) / y.std()
    z_outliers = df_pd[z_scores.abs() > 3].copy()

    ### Combined — flagged by either method
    all_outlier_dates = set(iqr_outliers["midday_day"]).union(set(z_outliers["midday_day"]))
    combined = df_pd[df_pd["midday_day"].isin(all_outlier_dates)].copy()
    combined["z_score"] = z_scores[df_pd["midday_day"].isin(all_outlier_dates)].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    ### Estimated avoidable deaths over time with outliers flagged
    axes[0].plot(df_pd["midday_day"], y, linewidth=0.5, color="steelblue", label="Y")
    axes[0].axhline(iqr_upper, color="red", linestyle="--", linewidth=1, label=f"IQR upper ({iqr_upper:.1f})")
    axes[0].axhline(iqr_lower, color="orange", linestyle="--", linewidth=1, label=f"IQR lower ({iqr_lower:.1f})")
    if len(combined) > 0:
        axes[0].scatter(combined["midday_day"], combined[TARGET],
                       color="red", zorder=5, s=40, label=f"{len(combined)} outliers")
    axes[0].set_title("Y over time with outliers flagged (IQR ± 3)")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("avoidable deaths / day")
    axes[0].legend()

    ### Z-scores over time
    axes[1].plot(df_pd["midday_day"], z_scores, linewidth=0.5, color="steelblue")
    axes[1].axhline(3, color="red", linestyle="--", linewidth=1, label="z = ±3")
    axes[1].axhline(-3, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    if len(z_outliers) > 0:
        axes[1].scatter(z_outliers["midday_day"], z_scores[z_scores.abs() > 3],
                       color="red", zorder=5, s=40)
    axes[1].set_title("Z-scores over time")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("z-score")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(EDA_DIR / "08-outliers.png", dpi=120)
    plt.close(fig)

    return {
        "iqr_lower_bound": float(iqr_lower),
        "iqr_upper_bound": float(iqr_upper),
        "n_iqr_outliers": len(iqr_outliers),
        "n_zscore_outliers": len(z_outliers),
        "n_combined_outliers": len(combined),
        "outlier_dates": [
            {
                "date": str(row["midday_day"].date()),
                "value": float(row[TARGET]),
                "z_score": float(row["z_score"])
            }
            for _, row in combined.sort_values(TARGET, ascending=False).iterrows()
        ],
    }

def write_report(missingness, y_dist, y_ts, time_cov, lag_summary, top_list, duration: float) -> None:
    out = EDA_DIR / "EDA-REPORT.md"

    # Loss-function decision logic
    skew = y_dist["skewness"]
    fraction_zero = y_dist["fraction_zero"]
    if fraction_zero > 0.1 or (skew > 1 and y_dist["min"] >= 0):
        loss = "tweedie (variance_power=1.1)"
        reason = "right-skewed and/or zero-inflated count — Tweedie is M5 winners' default"
    elif abs(skew) < 1:
        loss = "L2 / regression (Gaussian)"
        reason = "approximately symmetric"
    else:
        loss = "L2 with log1p target or Huber"
        reason = "skewed but not zero-inflated; transform first"

    lines = [
        "# EDA Report — Phase 1.2",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: `work/data/wide_daily.parquet` (930 × 350)",
        f"Runtime: {duration:.1f}s",
        "",
        "## 1. Missingness (`01-missingness.png`)",
        "",
        f"- Metrics total: **{missingness['n_metrics']}**",
        f"- Fully covered (0% missing): **{missingness['n_fully_covered']}**",
        f"- Partially covered: **{missingness['n_partial']}**",
        f"- All-empty (100% missing): **{missingness['n_empty']}**",
        f"- Mean missing rate: **{missingness['mean_missing_pct']:.1%}**",
        f"- Median missing rate: **{missingness['median_missing_pct']:.1%}**",
        "",
        "### 10 most-missing metrics",
        "",
    ]
    for m, p in missingness["top10_most_missing"].items():
        lines.append(f"- `{m}` — {p:.1%}")
    lines.extend([
        "",
        "### 10 best-covered metrics",
        "",
    ])
    for m, p in missingness["top10_best_covered"].items():
        lines.append(f"- `{m}` — {p:.1%}")

    lines.extend([
        "",
        "## 2. Target distribution (`02-y-distribution.png`)",
        "",
        f"- n = **{y_dist['n_valid']}** non-null daily observations",
        f"- mean = **{y_dist['mean']:.3f}**, median = **{y_dist['median']:.3f}**, std = **{y_dist['std']:.3f}**",
        f"- min = **{y_dist['min']:.3f}**, max = **{y_dist['max']:.3f}**",
        f"- p25 = **{y_dist['p25']:.3f}**, p75 = **{y_dist['p75']:.3f}**, p95 = **{y_dist['p95']:.3f}**",
        f"- skewness = **{y_dist['skewness']:.3f}**",
        f"- kurtosis = **{y_dist['kurtosis']:.3f}**",
        f"- fraction == 0 = **{y_dist['fraction_zero']:.2%}**",
        "",
        f"### Loss-function decision (driven by skew + zero-inflation)",
        "",
        f"- **Recommended loss: `{loss}`**",
        f"- Reason: {reason}",
        f"- Evidence anchor: `docs/literature/m5-winner-1-yj-stu.md` §4 Passage 5 (YJ_STU used Tweedie variance_power=1.1 for sparse non-negative counts).",
        "",
        "## 3. Target timeseries (`03-y-timeseries.png`)",
        "",
        f"- Peak Y: **{y_ts['max_y_value']:.2f}** on {y_ts['max_y_date']}",
        f"- Min  Y: **{y_ts['min_y_value']:.2f}** on {y_ts['min_y_date']}",
        "",
        "### Day-of-week means",
        "",
    ])
    for dow, m in y_ts["weekday_means"].items():
        lines.append(f"- {dow}: {m:.3f}")
    lines.extend([
        "",
        "## 4. Per-metric time coverage (`04-time-coverage.png`)",
        "",
        f"- Total metrics: **{time_cov['n_metrics']}**",
        f"- 95%+ time-covered: **{time_cov['n_full_coverage_95pct']}**",
        f"- Started after first day: **{time_cov['n_started_late']}**",
        f"- Ended before last day: **{time_cov['n_ended_early']}**",
        "",
        "Implication: late-starting metrics need careful lag handling — they cannot provide history before their start date. Some may need to be dropped if coverage is < 50%.",
        "",
        "## 5. Lag correlations with Y (`05-lag-correlations.csv`)",
        "",
        f"- {lag_summary['n_metric_lag_pairs']} (metric × lag) pairs tested across lags {lag_summary['lags_tested']}",
        f"- Top 30 by |Pearson| at any lag (full CSV in `05-lag-correlations.csv`):",
        "",
        "| Rank | Metric | Lag | Pearson | Spearman | n |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for i, r in enumerate(top_list, 1):
        lines.append(
            f"| {i} | `{r['metric'][:70]}` | {r['lag']} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['n']} |"
        )

    lines.extend([
        "",
        "### Sanity check against Q5.2 causal taxonomy",
        "",
        "Expected pattern (from `PROGRESS.md` Q5.2): top leading-indicator features should be dominated by NCtR variants, OPEL components, and ambulance / DTA / ED-pressure metrics. If the top-30 list above matches this, the causal taxonomy is validated empirically; if not, it's a flag to revisit.",
        "",
        "## 6. Implications for Phase 1.3 / 1.4",
        "",
        "1. **Loss function for LightGBM**: see §2 above.",
        "2. **Metrics to drop from primary feature set**: any with > 50% missingness AND no causal-role mapping in Phase 1.3.",
        "3. **Lag-feature horizons to prioritise**: lags where the top-30 list is most populated (likely 1, 7).",
        "4. **Day-of-week feature**: §3 weekday means show whether DoW matters; expect higher Y on Mondays (NHS weekday peak).",
        "",
    ])

    out.write_text("\n".join(lines))


def main():
    import time
    t0 = time.time()

    df = load_wide_daily()
    print(f"Loaded {df.shape}")

    print("  [1/8] missingness ...")
    missingness = plot_missingness(df)

    print("  [2/8] Y distribution ...")
    y_dist = plot_y_distribution(df)

    print("  [3/8] Y timeseries ...")
    y_ts = plot_y_timeseries(df)

    print("  [4/8] time coverage ...")
    time_cov = plot_time_coverage(df)

    print("  [5/8] lag correlations ...")
    lag_summary, top_list = lag_correlations(df)
    
    print("  [6/8] seasonality patterns ...")
    seasonality = plot_seasonality(df)
    
    print("  [7/8] autocorrelation ...")
    autocorr = plot_autocorrelation(df)
    print(f"  significant ACF lags  : {autocorr['significant_acf_lags']}")
    print(f"  significant PACF lags : {autocorr['significant_pacf_lags']}")
    print(f"  ACF at lag 1  : {autocorr['acf_lag1']:.3f}")
    print(f"  ACF at lag 7  : {autocorr['acf_lag7']:.3f}")
    print(f"  ACF at lag 28 : {autocorr['acf_lag28']:.3f}")
    
    print(" [8/8] outlier detection ...")
    outliers = plot_outliers(df)
    print(f"  outliers found  : {outliers['n_combined_outliers']}")
    print(f"  IQR bounds      : {outliers['iqr_lower_bound']:.1f} — {outliers['iqr_upper_bound']:.1f}")
    if outliers['outlier_dates']:
        print("  top outlier dates:")
        for o in outliers['outlier_dates'][:5]:
            print(f"    {o['date']}  Y={o['value']:.1f}  z={o['z_score']:.2f}")

    duration = time.time() - t0

    write_report(missingness, y_dist, y_ts, time_cov, lag_summary, top_list, duration)

    # Run log
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M") + "-eda"
    (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / run_id / "run.json").write_text(json.dumps({
        "missingness": missingness,
        "y_distribution": y_dist,
        "y_timeseries_meta": y_ts,
        "time_coverage": time_cov,
        "lag_summary": {k: v for k, v in lag_summary.items() if k != "top30_by_abs_pearson_at_any_lag"},
        "duration_seconds": duration,
    }, indent=2, default=str))

    print(f"=== EDA done in {duration:.1f}s ===")
    print(f"Report: {EDA_DIR / 'EDA-REPORT.md'}")


if __name__ == "__main__":
    main()
