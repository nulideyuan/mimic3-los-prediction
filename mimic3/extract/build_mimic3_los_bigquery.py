"""
MIMIC-3 LOS Dataset Builder — Leakage-Free (via BigQuery)
==========================================================
Produces under NLP/processed/:
  mimic3_los.csv                   — structured features (no discharge-coded leakage)
  notes_{category}.parquet         — one file per note category, first-48h window

Structured features (all available at / within first 48h of admission)
-----------------------------------------------------------------------
  Admission-time : anchor_age, gender_bin, admission_dayofweek,
                   admission_month, is_weekend, is_emergency,
                   insurance_cat, admission_location_cat
  First-48h labs : creatinine, wbc, hemoglobin, sodium, potassium,
                   bicarbonate, bun, glucose, lactate, platelets, inr
                   (first, mean, min, max, count per lab)
  First-48h vitals: heart_rate, sbp, dbp, spo2, resp_rate, temp_c, gcs_total
                   (mean, min, max per vital)
  ICU in 24h     : icu_within_24h flag (did patient reach ICU within first 24h)

Removed (all coded at discharge → leakage):
  Elixhauser ICD-9 comorbidities, num_diagnoses, num_procedures,
  num_prescriptions, total num_labs, num_transfers

Note extraction
---------------
  window  : admittime − 6 h  →  admittime + 48 h
  exclude : Discharge summary, iserror = 1
  output  : one parquet per category in processed/notes_{safe_name}.parquet
            → feed into extract_note_sections.py for section inspection

Run:
    source ~/my_python_env/venv/bin/activate
    python NLP/build_mimic3_los_bigquery.py
"""

import os
import numpy as np
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "eicu-494316"
CLINICAL   = "physionet-data.mimiciii_clinical"
NOTES      = "physionet-data.mimiciii_notes"
OUT        = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(OUT, exist_ok=True)

client = bigquery.Client(project=PROJECT_ID)


def bq(sql: str) -> pd.DataFrame:
    print(f"  → querying {len(sql)} chars ...")
    return client.query(sql).to_dataframe()


# ============================================================
# 1. Admissions + LOS
# ============================================================
print("1. Admissions ...")
adm = bq(f"""
SELECT
  subject_id, hadm_id,
  admittime, dischtime,
  admission_type,
  admission_location,
  insurance,
  hospital_expire_flag
FROM `{CLINICAL}.admissions`
""")
adm["admittime"] = pd.to_datetime(adm["admittime"])
adm["dischtime"] = pd.to_datetime(adm["dischtime"])
adm["los_days"]  = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400.0
adm = adm[(adm["los_days"] > 0) & (adm["los_days"] <= 365)].copy()
print(f"   {len(adm):,} admissions after LOS filter")

adm["admission_dayofweek"] = adm["admittime"].dt.weekday
adm["admission_month"]     = adm["admittime"].dt.month
adm["admission_year"]      = adm["admittime"].dt.year
adm["is_weekend"]          = adm["admittime"].dt.weekday.isin([5, 6]).astype(np.int8)

adm["is_emergency"] = (
    adm["admission_type"].str.upper().isin({"EMERGENCY", "URGENT"})
).astype(np.int8)

ins_map = {"Medicare": 0, "Medicaid": 1, "Private": 2, "Government": 2, "Self Pay": 3}
adm["insurance_cat"] = adm["insurance"].map(ins_map).fillna(3).astype(np.int8)

loc_order = adm["admission_location"].value_counts().index.tolist()
loc_map   = {loc: i for i, loc in enumerate(loc_order)}
adm["admission_location_cat"] = (
    adm["admission_location"].map(loc_map).fillna(len(loc_order)).astype(np.int8)
)

# ============================================================
# 2. Patients — age from DOB
# ============================================================
print("2. Patients ...")
pat = bq(f"""
SELECT subject_id, gender, dob
FROM `{CLINICAL}.patients`
""")
pat["dob"]        = pd.to_datetime(pat["dob"])
pat["gender_bin"] = pat["gender"].str.upper().map({"F": 0, "M": 1}).fillna(0).astype(np.int8)

adm = adm.merge(pat[["subject_id", "gender_bin", "dob"]], on="subject_id", how="left")
adm["anchor_age"] = ((adm["admittime"] - adm["dob"]).dt.days / 365.25).clip(0, 89).round(1)
adm = adm.drop(columns=["dob"])

# ============================================================
# 3. ICU within first 24h (leakage-free — reflects acuity at presentation)
# ============================================================
print("3. ICU within 24h ...")
icu_24h = bq(f"""
SELECT DISTINCT i.hadm_id, 1 AS icu_within_24h
FROM `{CLINICAL}.icustays` i
JOIN `{CLINICAL}.admissions` a USING (hadm_id)
WHERE TIMESTAMP_DIFF(i.intime, a.admittime, HOUR) <= 24
  AND TIMESTAMP_DIFF(i.intime, a.admittime, HOUR) >= 0
""")
icu_24h["icu_within_24h"] = icu_24h["icu_within_24h"].astype(np.int8)

# ============================================================
# 4. First-24h lab results  (first value only — leakage-free)
#    Drop lactate (>60% missing even at 48h).
# ============================================================
print("4. First-24h labs ...")

KEY_LABS = {
    "creatinine":  [50912],
    "wbc":         [51301],
    "hemoglobin":  [51222],
    "sodium":      [50983],
    "potassium":   [50971],
    "bicarbonate": [50882],
    "bun":         [51006],
    "glucose":     [50931],
    "platelets":   [51265],
    "inr":         [51237],
}
lab_id_str = ", ".join(str(i) for item in KEY_LABS.values() for i in item)

labs_raw = bq(f"""
SELECT l.hadm_id, l.itemid, l.valuenum, l.charttime
FROM `{CLINICAL}.labevents` AS l
INNER JOIN (
  SELECT hadm_id, admittime
  FROM `{CLINICAL}.admissions`
  WHERE TIMESTAMP_DIFF(dischtime, admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
) AS a ON l.hadm_id = a.hadm_id
WHERE l.itemid IN ({lab_id_str})
  AND l.valuenum IS NOT NULL
  AND l.valuenum > 0
  AND l.charttime >= TIMESTAMP_SUB(a.admittime, INTERVAL 6 HOUR)
  AND l.charttime <= TIMESTAMP_ADD(a.admittime, INTERVAL 24 HOUR)
""")
labs_raw["charttime"] = pd.to_datetime(labs_raw["charttime"])

itemid2lab = {i: name for name, ids in KEY_LABS.items() for i in ids}
labs_raw["lab"] = labs_raw["itemid"].map(itemid2lab)

valid_hadm = set(adm["hadm_id"])
labs_raw   = labs_raw[labs_raw["hadm_id"].isin(valid_hadm)]

labs_wide = pd.DataFrame({"hadm_id": adm["hadm_id"].values})
for lab_name, grp in labs_raw.groupby("lab"):
    first_val = (
        grp.sort_values("charttime")
           .groupby("hadm_id")["valuenum"]
           .first()
           .rename(f"lab_{lab_name}")
    )
    labs_wide = labs_wide.merge(first_val.reset_index(), on="hadm_id", how="left")

lab_cols = [c for c in labs_wide.columns if c != "hadm_id"]
cov = (labs_wide[lab_cols].notna().any(axis=1)).mean() * 100
print(f"   {len(lab_cols)} lab features  |  {cov:.1f}% of admissions have ≥1 lab")

# ============================================================
# 5. First-24h vital signs  (mean only — leakage-free)
#    Drop temp_c (>90% missing), gcs_total (>60% missing).
# ============================================================
print("5. First-24h vitals ...")

KEY_VITALS = {
    "heart_rate": [211, 220045],
    "sbp":        [51, 220179],
    "dbp":        [8368, 220180],
    "spo2":       [646, 220277],
    "resp_rate":  [618, 220210],
}
vital_id_str = ", ".join(str(i) for item in KEY_VITALS.values() for i in item)

vitals_raw = bq(f"""
SELECT c.hadm_id, c.itemid, c.valuenum, c.charttime
FROM `{CLINICAL}.chartevents` AS c
INNER JOIN (
  SELECT hadm_id, admittime
  FROM `{CLINICAL}.admissions`
  WHERE TIMESTAMP_DIFF(dischtime, admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
) AS a ON c.hadm_id = a.hadm_id
WHERE c.itemid IN ({vital_id_str})
  AND c.valuenum IS NOT NULL
  AND c.valuenum > 0
  AND c.charttime >= TIMESTAMP_SUB(a.admittime, INTERVAL 6 HOUR)
  AND c.charttime <= TIMESTAMP_ADD(a.admittime, INTERVAL 24 HOUR)
""")

itemid2vital  = {i: name for name, ids in KEY_VITALS.items() for i in ids}
vitals_raw["vital"] = vitals_raw["itemid"].map(itemid2vital)
vitals_raw = vitals_raw[vitals_raw["hadm_id"].isin(valid_hadm)]

vitals_wide = pd.DataFrame({"hadm_id": adm["hadm_id"].values})
for vital_name, grp in vitals_raw.groupby("vital"):
    mean_val = (
        grp.groupby("hadm_id")["valuenum"]
           .mean()
           .rename(f"vital_{vital_name}")
    )
    vitals_wide = vitals_wide.merge(mean_val.reset_index(), on="hadm_id", how="left")

vital_cols = [c for c in vitals_wide.columns if c != "hadm_id"]
cov = (vitals_wide[vital_cols].notna().any(axis=1)).mean() * 100
print(f"   {len(vital_cols)} vital features  |  {cov:.1f}% of admissions have ≥1 vital")

# ============================================================
# 6. Prescriptions within first 24h (count of distinct drugs)
# ============================================================
print("6. Prescriptions within 24h ...")
rx_24h = bq(f"""
SELECT p.hadm_id, COUNT(DISTINCT p.drug) AS rx_count_24h
FROM `{CLINICAL}.prescriptions` p
INNER JOIN (
  SELECT hadm_id, admittime
  FROM `{CLINICAL}.admissions`
  WHERE TIMESTAMP_DIFF(dischtime, admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
) a ON p.hadm_id = a.hadm_id
WHERE DATE(p.startdate) BETWEEN DATE(TIMESTAMP_SUB(a.admittime, INTERVAL 6 HOUR))
                             AND DATE(TIMESTAMP_ADD(a.admittime, INTERVAL 24 HOUR))
  AND p.drug IS NOT NULL
GROUP BY p.hadm_id
""")
print(f"   {rx_24h['hadm_id'].nunique():,} admissions have ≥1 prescription in 24h")

# ============================================================
# 7. Prior admissions count (history of hospitalisation)
#    Only admissions fully discharged before current admittime.
# ============================================================
print("7. Prior admissions ...")
prior_adm = bq(f"""
SELECT a.hadm_id,
       COUNT(p.hadm_id) AS prior_admissions
FROM `{CLINICAL}.admissions` a
LEFT JOIN `{CLINICAL}.admissions` p
  ON  p.subject_id  = a.subject_id
  AND p.dischtime   < a.admittime
  AND p.hadm_id    != a.hadm_id
WHERE TIMESTAMP_DIFF(a.dischtime, a.admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
GROUP BY a.hadm_id
""")
print(f"   {len(prior_adm):,} admissions with prior history computed")

# ============================================================
# 8. Merge all structured features
# ============================================================
print("8. Merging ...")
df = adm.drop(
    columns=["admission_type", "admission_location", "insurance",
             "dischtime", "subject_id", "hospital_expire_flag"],
    errors="ignore",
)

for tbl in [icu_24h, labs_wide, vitals_wide, rx_24h, prior_adm]:
    df = df.merge(tbl, on="hadm_id", how="left")

df["icu_within_24h"]   = df["icu_within_24h"].fillna(0).astype(np.int8)
df["rx_count_24h"]     = df["rx_count_24h"].fillna(0).astype(np.int16)
df["prior_admissions"] = df["prior_admissions"].fillna(0).astype(np.int16)
df = df.sort_values("admittime").reset_index(drop=True)

struct_path = os.path.join(OUT, "mimic3_los.csv")
df.to_csv(struct_path, index=False)
print(f"   Saved → {struct_path}  {df.shape[0]:,} rows × {df.shape[1]} cols")
feat_cols = [c for c in df.columns if c not in ("hadm_id", "admittime", "los_days")]
print(f"   Features: {len(feat_cols)}  |  Missing cells: {df[feat_cols].isna().sum().sum():,}")

# ============================================================
# 7a. Non-physician notes — 48h window only
#     Dynamic content: nursing observations, radiology findings,
#     ECG/echo reports.  Time-gating is essential here.
#     Notes with NULL charttime are dropped (cannot place in window).
# ============================================================
print("7a. Non-physician notes (48h window, charttime NOT NULL) ...")

notes_dynamic = bq(f"""
SELECT
  n.hadm_id,
  n.charttime,
  n.category,
  SUBSTR(n.text, 1, 12000) AS text
FROM `{NOTES}.noteevents` AS n
INNER JOIN (
  SELECT hadm_id, admittime
  FROM `{CLINICAL}.admissions`
  WHERE TIMESTAMP_DIFF(dischtime, admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
) AS a ON n.hadm_id = a.hadm_id
WHERE
  n.hadm_id IS NOT NULL
  AND COALESCE(n.iserror, 0) != 1
  AND LOWER(TRIM(n.category)) NOT IN ('discharge summary', 'physician')
  AND n.charttime IS NOT NULL
  AND n.charttime >= TIMESTAMP_SUB(a.admittime, INTERVAL 6 HOUR)
  AND n.charttime <= TIMESTAMP_ADD(a.admittime, INTERVAL 48 HOUR)
""")
notes_dynamic["charttime"] = pd.to_datetime(notes_dynamic["charttime"])
notes_dynamic = notes_dynamic[notes_dynamic["hadm_id"].isin(set(df["hadm_id"]))].copy()
notes_dynamic = notes_dynamic.sort_values(["hadm_id", "charttime"]).reset_index(drop=True)

print(f"   {len(notes_dynamic):,} notes | {notes_dynamic['hadm_id'].nunique():,} admissions")
for cat, grp in notes_dynamic.groupby("category"):
    safe = cat.strip().lower().replace("/", "_").replace(" ", "_")
    path = os.path.join(OUT, f"notes_{safe}.parquet")
    grp.reset_index(drop=True).to_parquet(path, index=False, compression="snappy")
    n_adm = grp["hadm_id"].nunique()
    pct   = n_adm / len(df) * 100
    print(f"   {cat:<25} {len(grp):>7,} notes  {n_adm:>6,} adm ({pct:.1f}%)")

# ============================================================
# 7b. Physician notes — NO time filter
#     We take every physician note for the admission regardless of
#     when it was written.  The section extractor (extract_note_sections.py)
#     then routes each section to the right leakage bucket:
#
#       STABLE  (no time filter used here):
#         CC, HPI, PMH, Past Surgical Hx, Family Hx,
#         Social Hx, Allergies, Medications
#
#       DYNAMIC (section extractor re-applies 48h filter internally
#                using the admittime column saved below):
#         Physical Exam, Assessment & Plan, ROS, Labs/Results
#
#     Notes with NULL charttime are included — they can still contribute
#     stable sections (PMH etc.) even without a precise timestamp.
# ============================================================
print("\n7b. Physician notes (all-time, no charttime filter) ...")

physician_all = bq(f"""
SELECT
  n.hadm_id,
  n.charttime,
  n.category,
  SUBSTR(n.text, 1, 12000) AS text,
  a.admittime                          -- kept so section extractor can apply 48h gate
FROM `{NOTES}.noteevents` AS n
INNER JOIN (
  SELECT hadm_id, admittime
  FROM `{CLINICAL}.admissions`
  WHERE TIMESTAMP_DIFF(dischtime, admittime, SECOND) / 86400.0 BETWEEN 0 AND 365
) AS a ON n.hadm_id = a.hadm_id
WHERE
  n.hadm_id IS NOT NULL
  AND COALESCE(n.iserror, 0) != 1
  AND LOWER(TRIM(n.category)) = 'physician'
""")
physician_all["charttime"] = pd.to_datetime(physician_all["charttime"])
physician_all["admittime"] = pd.to_datetime(physician_all["admittime"])
physician_all = physician_all[physician_all["hadm_id"].isin(set(df["hadm_id"]))].copy()
physician_all = physician_all.sort_values(["hadm_id", "charttime"],
                                          na_position="last").reset_index(drop=True)

path_phys = os.path.join(OUT, "notes_physician_all.parquet")
physician_all.to_parquet(path_phys, index=False, compression="snappy")
n_adm = physician_all["hadm_id"].nunique()
pct   = n_adm / len(df) * 100
print(f"   {len(physician_all):,} notes | {n_adm:,} admissions ({pct:.1f}%) → {path_phys}")
