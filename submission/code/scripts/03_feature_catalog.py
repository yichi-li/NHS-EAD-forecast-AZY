"""03 — Build the feature catalog by applying Q5.2 causal classification.

Output:
  work/outputs/features/feature_catalog.parquet
  work/outputs/features/feature_catalog.md   (human-readable)
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_wide_daily  # noqa: E402
from src.feature_catalog import build_catalog  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "work" / "outputs" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

### this script calls the catalog from src/feature_catalog.py
### code for regex pattern matching is found there

def main():
    df = load_wide_daily()
    catalog = build_catalog(df.columns)
    n_cols = len(catalog)

    # Save parquet
    parquet_path = OUT_DIR / "feature_catalog.parquet"
    catalog.write_parquet(parquet_path)

    # Group counts
    by_role = catalog.group_by("causal_role").agg(pl.len().alias("n")).sort("n", descending=True)
    by_opel = catalog.group_by("opel_category").agg(pl.len().alias("n")).sort("n", descending=True)
    by_howlett = catalog.group_by("howlett_equivalent").agg(pl.len().alias("n")).sort("n", descending=True)

    print(f"Cataloged {n_cols} columns")
    print()
    print("By causal_role:")
    print(by_role)
    print()
    print("By opel_category:")
    print(by_opel)
    print()
    print("By howlett_equivalent:")
    print(by_howlett)

    # Markdown report
    md_path = OUT_DIR / "feature_catalog.md"
    lines = [
        f"# Feature Catalog — Phase 1.3",
        f"",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source columns: `work/data/wide_daily.parquet` ({n_cols} non-date columns)",
        f"Classification logic: `work/src/feature_catalog.py` (deterministic regex over column name).",
        f"",
        f"## Evidence anchors",
        f"",
        f"- **Causal-role 3-way split** ← `PROGRESS.md` 2026-05-19 \"Q5.2: Boarding upstream / concurrent / downstream\"",
        f"- **Howlett-direct mapping** ← `docs/literature/howlett-2026.md` §5",
        f"- **OPEL 10 acute parameters** ← `docs/literature/nhs-opel.md` §4 [6]",
        f"- **OPEL community/MH/111 parameters** ← `docs/literature/nhs-opel.md` §4 [10][11][12]",
        f"",
        f"## Summary counts",
        f"",
        f"### By causal_role",
        f"",
    ]
    for row in by_role.iter_rows(named=True):
        lines.append(f"- **{row['causal_role']}**: {row['n']}")

    lines.extend([
        f"",
        f"### By opel_category",
        f"",
    ])
    for row in by_opel.iter_rows(named=True):
        lines.append(f"- **{row['opel_category']}**: {row['n']}")

    lines.extend([
        f"",
        f"### By howlett_equivalent",
        f"",
    ])
    for row in by_howlett.iter_rows(named=True):
        lines.append(f"- **{row['howlett_equivalent']}**: {row['n']}")

    # Per-role example column listing (first 15 in each)
    lines.extend([
        f"",
        f"## Sample columns per causal_role (15 each)",
        f"",
    ])
    for role in ["upstream", "concurrent", "downstream", "other", "target"]:
        sub = catalog.filter(pl.col("causal_role") == role)
        lines.append(f"### {role} ({len(sub)} cols)")
        lines.append("")
        for c in sub["column"].to_list()[:15]:
            lines.append(f"- `{c}`")
        if len(sub) > 15:
            lines.append(f"- ... and {len(sub) - 15} more")
        lines.append("")

    # Flag the 'other' bucket — these need manual review
    other = catalog.filter(pl.col("causal_role") == "other")
    if len(other) > 0:
        lines.extend([
            f"## ⚠️  Columns classified as `other` (review)",
            f"",
            f"These {len(other)} columns did not match any of the upstream / concurrent / downstream patterns. They will still be available as features (categorical / numerical), but they don't have a causal role assigned, so feature-ablation by causal-role analysis will exclude them.",
            f"",
        ])
        for c in other["column"].to_list():
            lines.append(f"- `{c}`")
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nWrote:")
    print(f"  {parquet_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
