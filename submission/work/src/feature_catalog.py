"""Classify each of the 349 dataset columns by causal role / OPEL category /
Howlett relevance, based on the Q5.2 + Q1 evidence chains in PROGRESS.md.

The classification is deterministic (regex over the column name = `{metric} - {coverage}`)
so it can be re-derived and audited at any time.

Mapping rules — evidence anchors in `docs/literature/`:

UPSTREAM of boarding (Howlett Methods Stats p.2: NCtR = exit-block proxy)
- NCtR variants (any "*NCtR*" pattern, all hospitals & pathways)
- DTA counts ("No. of DTAs", "No. of DTAs (> 8hrs)")
- ED arrivals ("New Arrivals in Last Hour", "Ambulances Conveyed", "Ambulance Queue")
- Escalation beds / IPC closures ("Escalation beds open", "Beds occupied by long-stay")
- Community pathway P1/P2/P3 capacity (Sirona / NHS @ Home / Closed TOC)
- 999 / Severnside upstream demand (calls received, conveyed, referred)
- DtA referrals, waiting lists, slots booked

CONCURRENT (system state at boarding time)
- OPEL composite + Aggregated NHSE OPEL Score
- Majors / resus occupancy
- Patients in A&E (total population)
- Time to triage / assessment / treatment
- G&A bed occupancy
- NHS 111 / Severnside call handling metrics

DOWNSTREAM (Howlett Table 2 p.6 — boarding CAUSES these)
- Ambulance response time (Cat 1-4) — mean & 90th percentile
- Ambulance handover times (15/30/60 min thresholds, last hour & since midnight)
- Handover to Clear / Handover Time Lost
- 4hr Breach Performance, Total Breaches Since Midnight
- "% of patients spending >12 hours in ED"

OTHER (target itself, staffing reductions, etc.) — not part of the 3-way causal taxonomy
- estimated_avoidable_deaths (target)
- Sirona staffing reductions
- Mental-health specific
- Capacity Available Slots Today (capacity offer, not pressure)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
import polars as pl


@dataclass
class FeatureClass:
    causal_role: str         # upstream / concurrent / downstream / other / target
    opel_category: str       # acute / community / mental_health / nhs_111 / not_opel
    howlett_eq: str          # direct / proxy / none
    leading_indicator: bool  # convenience flag = upstream OR confirmed-leading by EDA


# --- Pattern dictionaries (case-insensitive substring matches over the full column name)

UPSTREAM_PATTERNS = [
    r"\bNCtR\b",
    r"No\. of DTAs",
    r"DtA P[123]",
    r"DtA SBRU",
    r"DtA Community",
    r"Closed TOC",
    r"Delayed TOC",
    r"P[123] (Slots|Beds|Acute|Referrals|Caseload|NCtR|Admissions|Patients|\(Available\))",
    r"New Arrivals in Last Hour",
    r"Ambulances? Conveyed to Hospital",
    r"Ambulances? En Route",
    r"Ambulance Queue",
    r"Ambulances En Route to Hospital",
    r"Escalation beds open",
    r"Beds occupied by long-stay patients",
    r"Number of Active Calls on the 999",
    r"Number of Waiting Calls on the 999",
    r"Referred to 999",
    r"Calls Received",
    r"Calls Offered",
    r"Calls Answered",
    r"Calls Triaged",
    r"Calls Abandoned",
    r"Referrals Received",
    r"Number of Admissions",
    r"Number of GP expected diverts",
    r"\(Severnside\) CAS Referred to ED",
    r"\(Severnside\) Referred to ED",
    r"NHS @ Home",
    r"Section 136 Beds",
    r"Urgent Referrals",
    r"HCP Calls",
    r"IUC ",
    r"Number of Home Visits",
    r"Number of Treatment Centre",
    r"Cases on IUC CAS Queue",
    r"Deficit of Discharges",
    r"Number of Medical Outliers",
    r"Number of Surgical Outliers",
    r"Number of Cardiac Outliers",
    r"Number of Outliers",
    r"Cohorting & Reverse Queue",
    r"OOT Internal Placements",
    r"empty beds on assessment units",
    r"critical care beds available",
    r"% of beds occupied by patients with NCtR",
    r"% of open beds that are escalation beds",
]
# Q6.1 fix 2026-05-25: P0 Discharges / Complex Discharges / SWASFT incidents
# moved to CONCURRENT patterns below — they are observed system-state measures
# (events that already happened), not causally upstream of boarding pressure.

CONCURRENT_PATTERNS = [
    r"\bOPEL\b",
    r"Automated OPEL",
    r"Aggregated NHSE OPEL Score",
    r"Patients in A&E",
    r"Majors patient count",
    r"Minors patient count",
    r"Resuscitation Capacity",
    r"Majors and resuscitation occupancy",
    r"Average Time to Triage",
    r"Average Time to Assessment",
    r"Median time to treatment",
    r"G&A Bed occupancy",
    r"G&A beds, core stock open",
    r"adult critical care beds occupied",
    r"Critical Care beds",
    r"neonatal critical care beds",
    r"Bed Occupancy",
    r"ED all-type attendance variation",
    r"Calls Offered in Last 15",
    r"Calls Answered in Last 15",
    r"Calls Abandoned",
    r"Average Speed to Answer",
    r"Average Handling Time",
    r"Answered in 60 seconds",
    r"Longest Wait",
    r"DOS Status at 10am",
    r"Clinical Escalation Level",
    r"% Staffing Reduction",
    r"Intensive Caseload",
    r"Red Nursing schedules",
    r"Mason Section 136",
    r"Proportion of Calls Since Midnight",
    r"A&E attends - paediatrics",
    r"Number of Discharges",
    r"Number of cases on",
    r"Capacity \(Available Slots Today\)",
    # Q6.1 fix 2026-05-25 — these are observed system-state, not causally upstream:
    r"\bP0 Discharges\b",
    r"\bComplex Discharges\b",
    r"\(SWASFT\) Number of",
]

DOWNSTREAM_PATTERNS = [
    r"^Category [1234] - BNSSG (Mean|90th) Response",
    r"\(Severnside\) Referred to 999 - C[1-4] Ambulances",
    r"Ambulance Handovers? \d+",
    r"Handover to Clear",
    r"Handover Time Lost",
    r"4hr Breach Performance",
    r"Total Breaches Since Midnight",
    r"\bED all-type 4-hour performance",
    r"% of patients spending >12 hours in ED",
    r"A&E Discharges in Last Hour",
]

TARGET_PATTERNS = [r"^estimated_avoidable_deaths"]

# Howlett's named covariates (Methods Stats p.2 + Table 1)
HOWLETT_DIRECT_PATTERNS = [
    r"No\. of DTAs",          # = boarding count
    r"No\. of DTAs \(> 8hrs\)",  # = long-boarder count, near-equivalent to boarding-time exposure
    r"\bNCtR\b",              # explicit Howlett term, "exit block proxy"
    r"Patients in A&E",       # = ED occupancy
    r"Majors patient count",  # = majors occupancy (Howlett uses %-of-capacity)
    r"^Category [1234]",      # ambulance response cat
    r"Ambulance Handover",
    r"Handover to Clear",
    r"A&E Discharges in Last Hour",
    r"4hr Breach Performance",
    r"% of patients spending >12 hours in ED",
]

# OPEL official 10 acute parameters (docs/literature/nhs-opel.md §4 [6])
OPEL_ACUTE_PATTERNS = [
    r"Ambulance Handover",
    r"4hr Breach Performance",
    r"^ED all-type 4-hour performance",
    r"Majors and resuscitation occupancy",
    r"Median time to treatment",
    r"% of patients spending >12 hours in ED",
    r"G&A Bed occupancy",
    r"% of beds occupied by patients with NCtR",
    r"Number of Discharges",
    r"Escalation beds open",
    r"% of open beds that are escalation beds",
    r"\bOPEL\b",
    r"Automated OPEL",
    r"Aggregated NHSE OPEL Score",
]

OPEL_COMMUNITY_PATTERNS = [
    r"DtA Community",
    r"DtA P[123]",
    r"NHS @ Home",
    r"P[123]",
    r"Closed TOC",
    r"Delayed TOC",
    r"OPEL - .*Council",
    r"OPEL - .*Adult Health",
    r"OPEL - Sirona",
    r"Caseload",
    r"UCR",
]

OPEL_MH_PATTERNS = [r"\bAWP\b", r"Section 136"]

OPEL_111_PATTERNS = [
    r"\(Severnside\)",
    r"IUC",
    r"Calls (Offered|Answered|Abandoned|Triaged|Received)",
    r"Number of Home Visits",
    r"Number of Treatment Centre",
    r"OPEL - SevernSide",
]


def _matches_any(name: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, name, flags=re.IGNORECASE):
            return True
    return False


def classify_column(col_name: str) -> FeatureClass:
    """Return causal_role / opel_category / howlett_eq / leading_indicator for one column."""

    if _matches_any(col_name, TARGET_PATTERNS):
        return FeatureClass("target", "not_opel", "none", False)

    if _matches_any(col_name, DOWNSTREAM_PATTERNS):
        role = "downstream"
    elif _matches_any(col_name, UPSTREAM_PATTERNS):
        role = "upstream"
    elif _matches_any(col_name, CONCURRENT_PATTERNS):
        role = "concurrent"
    else:
        role = "other"

    if _matches_any(col_name, OPEL_ACUTE_PATTERNS):
        opel_cat = "acute"
    elif _matches_any(col_name, OPEL_COMMUNITY_PATTERNS):
        opel_cat = "community"
    elif _matches_any(col_name, OPEL_MH_PATTERNS):
        opel_cat = "mental_health"
    elif _matches_any(col_name, OPEL_111_PATTERNS):
        opel_cat = "nhs_111"
    else:
        opel_cat = "not_opel"

    if _matches_any(col_name, HOWLETT_DIRECT_PATTERNS):
        howlett = "direct"
    elif role in ("upstream", "downstream"):
        howlett = "proxy"
    else:
        howlett = "none"

    leading = role == "upstream"
    return FeatureClass(role, opel_cat, howlett, leading)


def build_catalog(columns: list[str]) -> pl.DataFrame:
    """Apply classify_column to every column name; return a polars DataFrame."""
    rows = []
    for c in columns:
        if c == "midday_day":
            continue
        cls = classify_column(c)
        # Parse "metric - coverage" if possible
        if " - " in c:
            metric, coverage = c.rsplit(" - ", 1)
        else:
            metric, coverage = c, ""
        rows.append({
            "column": c,
            "metric_name": metric,
            "coverage_label": coverage,
            "causal_role": cls.causal_role,
            "opel_category": cls.opel_category,
            "howlett_equivalent": cls.howlett_eq,
            "leading_indicator": cls.leading_indicator,
        })
    return pl.DataFrame(rows)
