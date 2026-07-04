# Feature Catalog — Phase 1.3

Generated: 2026-07-03T13:00:19
Source columns: `work/data/wide_daily.parquet` (352 non-date columns)
Classification logic: `work/src/feature_catalog.py` (deterministic regex over column name).

## Evidence anchors

- **Causal-role 3-way split** ← `PROGRESS.md` 2026-05-19 "Q5.2: Boarding upstream / concurrent / downstream"
- **Howlett-direct mapping** ← `docs/literature/howlett-2026.md` §5
- **OPEL 10 acute parameters** ← `docs/literature/nhs-opel.md` §4 [6]
- **OPEL community/MH/111 parameters** ← `docs/literature/nhs-opel.md` §4 [10][11][12]

## Summary counts

### By causal_role

- **upstream**: 167
- **concurrent**: 106
- **downstream**: 78
- **target**: 1

### By opel_category

- **not_opel**: 177
- **acute**: 72
- **community**: 70
- **nhs_111**: 28
- **mental_health**: 5

### By howlett_equivalent

- **proxy**: 142
- **direct**: 111
- **none**: 99

## Sample columns per causal_role (15 each)

### upstream (167 cols)

- `% of beds occupied by patients with NCtR - BRI`
- `% of beds occupied by patients with NCtR - NBT`
- `% of beds occupied by patients with NCtR - WGH`
- `% of open beds that are escalation beds - BRI`
- `% of open beds that are escalation beds - WGH`
- `(Severnside) CAS Referred to ED - SevernSide`
- `(Severnside) Calls Answered - SevernSide`
- `(Severnside) Calls Answered Within 60 Seconds - SevernSide`
- `(Severnside) Calls Received - SevernSide`
- `(Severnside) Calls Triaged - SevernSide`
- `(Severnside) HCP Calls Answered - SevernSide`
- `(Severnside) Referred to ED - SevernSide`
- `Ambulance Queue - BRI`
- `Ambulance Queue - NBT`
- `Ambulance Queue - WGH`
- ... and 152 more

### concurrent (106 cols)

- `(SWASFT) Number of HCP Incidents - SWASFT`
- `(SWASFT) Number of NHS 111 Incidents - SWASFT`
- `A&E attends - paediatrics - BRI`
- `A&E attends - paediatrics - NBT`
- `A&E attends - paediatrics - WGH`
- `Adult critical care beds occupied - NBT`
- `Aggregated NHSE OPEL Score - BRI`
- `Aggregated NHSE OPEL Score - NBT`
- `Aggregated NHSE OPEL Score - WGH`
- `Answered in 60 seconds Since Midnight that Received (%) - SevernSide`
- `Answered in 60 seconds Since Midnight that Received - SevernSide`
- `Answered in 60 seconds in Last 15 mins (%) - SevernSide`
- `Answered in 60 seconds in Last 15 mins - SevernSide`
- `Automated OPEL - BRI`
- `Automated OPEL - NBT`
- ... and 91 more

### downstream (78 cols)

- `% of patients spending >12 hours in ED - BRI`
- `% of patients spending >12 hours in ED - NBT`
- `% of patients spending >12 hours in ED - WGH`
- `(Severnside) Referred to 999 - C1 Ambulances - SevernSide`
- `(Severnside) Referred to 999 - C2 Ambulances - SevernSide`
- `(Severnside) Referred to 999 - C3 Ambulances - SevernSide`
- `(Severnside) Referred to 999 - C4 Ambulances - SevernSide`
- `4hr Breach Performance - BRI`
- `4hr Breach Performance - BRI (Children's)`
- `4hr Breach Performance - NBT`
- `4hr Breach Performance - WGH`
- `A&E Discharges in Last Hour - BRI`
- `A&E Discharges in Last Hour - BRI (Children's)`
- `A&E Discharges in Last Hour - NBT`
- `A&E Discharges in Last Hour - WGH`
- ... and 63 more

### other (0 cols)


### target (1 cols)

- `estimated_avoidable_deaths`
