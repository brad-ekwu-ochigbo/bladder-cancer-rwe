"""
Paper 1 - Script 1: Cohort Identification and Data Cleaning
===========================================================
Study: Stage at Diagnosis, Surgical Treatment Utilization, and Survival
       Outcomes in MIBC/Metastatic Bladder Cancer Before and After ICI
       Availability: A SEER Population-Based Analysis, 2004-2021

Input:  Raw SEER case listing CSV exported from SEER*Stat
Output: cleaned_cohort.csv — analysis-ready dataset

SEER*Stat Export Instructions:
  1. Open SEER*Stat > Case Listing Session
  2. Database: Incidence SEER Research Data, 17 Registries, Nov 2023 Sub (2000-2021)
  3. Selection: Primary Site = C670-C679 (bladder)
  4. Variables to include (all listed in this script's COLUMN_MAP)
  5. Export as CSV (semicolon or comma delimited)
  6. Save as: data/seer_raw.csv
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────

RAW_DATA_PATH  = "data/seer_raw.csv"
OUTPUT_PATH    = "outputs/cleaned_cohort.csv"
LOG_PATH       = "outputs/cohort_build_log.txt"

STUDY_START    = 2004    # Treatment variables available from 2004
STUDY_END      = 2021
ICI_YEAR       = 2016    # FDA approval year — excluded as transition year
PRE_ICI_END    = 2015
POST_ICI_START = 2017

# ICD-O-3 histology codes for urothelial (transitional cell) carcinoma
UROTHELIAL_HIST = [str(h) for h in range(8120, 8131)]  # 8120-8130

# Primary site codes for bladder C67.x
BLADDER_SITES = [f"C67{str(i)}" for i in range(10)]  # C670-C679

# ─── COLUMN MAP ──────────────────────────────────────────────────────────────
# Maps raw SEER*Stat column names to clean working names.
# Adjust left side to match your actual SEER*Stat export headers.

COLUMN_MAP = {
    "Patient ID":                          "patient_id",
    "Year of diagnosis":                   "year_dx",
    "Age at diagnosis":                    "age_dx",
    "Sex":                                 "sex",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": "race_eth",
    "Marital status at diagnosis":         "marital_status",
    "SEER registry":                       "seer_registry",
    "Rural-Urban Continuum Code":          "rucc",
    "Median household income quartile (2006-2010)": "income_quartile",
    "Primary Site":                        "primary_site",
    "Histologic Type ICD-O-3":             "histology",
    "Grade":                               "grade",
    "Derived AJCC Stage Group, 7th ed (2010-2015)": "ajcc_stage",
    "Combined Summary Stage (2004+)":      "summary_stage",
    "RX Summ--Surg Prim Site":             "surgery_code",
    "Radiation recode":                    "radiation",
    "Survival months":                     "survival_months",
    "Vital status recode (study cutoff used)": "vital_status",
    "SEER cause-specific death classification": "css_death",
    "Sequence number":                     "sequence_number",
}

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────

log("=" * 65)
log("PAPER 1 — COHORT IDENTIFICATION")
log("=" * 65)

if not os.path.exists(RAW_DATA_PATH):
    log(f"\nERROR: Raw data file not found at '{RAW_DATA_PATH}'")
    log("Please export your SEER case listing to data/seer_raw.csv")
    log("See script header for SEER*Stat export instructions.")
    exit(1)

log(f"\nLoading raw data from: {RAW_DATA_PATH}")
df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)
log(f"  Raw rows loaded:    {len(df_raw):,}")
log(f"  Raw columns:        {len(df_raw.columns)}")

# ─── RENAME COLUMNS ──────────────────────────────────────────────────────────

# Rename only columns that exist in the file (graceful handling)
rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df_raw.columns}
missing_cols = [k for k in COLUMN_MAP if k not in df_raw.columns]
df = df_raw.rename(columns=rename_map)

if missing_cols:
    log(f"\nWARNING: These expected SEER columns were not found:")
    for c in missing_cols:
        log(f"  - {c}")
    log("  Adjust COLUMN_MAP in this script to match your SEER*Stat export.")

log(f"\nColumns successfully mapped: {len(rename_map)}")

# ─── STEP 1: RESTRICT TO STUDY PERIOD ────────────────────────────────────────

log("\n--- Step 1: Study period restriction ---")
n_before = len(df)
df["year_dx"] = pd.to_numeric(df["year_dx"], errors="coerce")
df = df[df["year_dx"].between(STUDY_START, STUDY_END)]
log(f"  Kept 2004-2021:     {len(df):,}  (excluded {n_before - len(df):,})")

# ─── STEP 2: BLADDER PRIMARY SITE ────────────────────────────────────────────

log("\n--- Step 2: Primary site filter (C67.x bladder) ---")
n_before = len(df)
if "primary_site" in df.columns:
    df["primary_site"] = df["primary_site"].astype(str).str.strip().str.upper()
    df = df[df["primary_site"].str.startswith("C67")]
    log(f"  After site filter:  {len(df):,}  (excluded {n_before - len(df):,})")
else:
    log("  WARNING: primary_site column not found — skipping site filter")

# ─── STEP 3: UROTHELIAL HISTOLOGY ────────────────────────────────────────────

log("\n--- Step 3: Urothelial histology (8120-8130) ---")
n_before = len(df)
if "histology" in df.columns:
    df["histology"] = df["histology"].astype(str).str.strip()
    df["is_urothelial"] = df["histology"].isin(UROTHELIAL_HIST)
    # Flag non-urothelial for sensitivity analysis
    df_non_uro = df[~df["is_urothelial"]].copy()
    df = df[df["is_urothelial"]].copy()
    log(f"  Urothelial cases:   {len(df):,}")
    log(f"  Non-urothelial:     {len(df_non_uro):,}  (saved for sensitivity analysis)")
    df_non_uro.to_csv("outputs/sensitivity_non_urothelial.csv", index=False)
else:
    log("  WARNING: histology column not found — skipping histology filter")

# ─── STEP 4: MUSCLE-INVASIVE / METASTATIC ONLY ───────────────────────────────

log("\n--- Step 4: Restrict to MIBC/metastatic (AJCC stage II-IV) ---")
n_before = len(df)

if "ajcc_stage" in df.columns:
    df["ajcc_stage"] = df["ajcc_stage"].astype(str).str.strip().str.upper()
    mibc_stages = ["II", "IIA", "IIB", "IIC", "III", "IIIA", "IIIB",
                   "IV", "IVA", "IVB", "2", "3", "4"]
    df["is_mibc"] = df["ajcc_stage"].isin(mibc_stages)

    # Fallback: use summary stage if AJCC not available
    if df["is_mibc"].sum() < 100 and "summary_stage" in df.columns:
        log("  Low AJCC match — falling back to SEER Summary Stage")
        df["summary_stage"] = df["summary_stage"].astype(str).str.strip()
        df["is_mibc"] = df["summary_stage"].isin(["Regional", "Distant",
                                                    "2", "3", "4", "5"])

    df_stage1 = df[~df["is_mibc"]].copy()
    df = df[df["is_mibc"]].copy()
    log(f"  MIBC/metastatic:    {len(df):,}")
    log(f"  Stage I / localized:{len(df_stage1):,}  (saved for sensitivity analysis)")
    df_stage1.to_csv("outputs/sensitivity_stage1.csv", index=False)
else:
    log("  WARNING: ajcc_stage column not found — skipping stage filter")

# ─── STEP 5: EXCLUDE UNKNOWN STAGE ───────────────────────────────────────────

log("\n--- Step 5: Exclude unknown stage ---")
n_before = len(df)
unknown_stage_vals = ["Unknown", "UNK", "88", "99", "nan", ""]
if "ajcc_stage" in df.columns:
    df = df[~df["ajcc_stage"].isin(unknown_stage_vals)]
log(f"  After unknown stage exclusion: {len(df):,}  (excluded {n_before - len(df):,})")

# ─── STEP 6: AGE >= 18 ───────────────────────────────────────────────────────

log("\n--- Step 6: Age >= 18 at diagnosis ---")
n_before = len(df)
df["age_dx"] = pd.to_numeric(df["age_dx"], errors="coerce")
df = df[df["age_dx"] >= 18]
log(f"  Age >= 18:          {len(df):,}  (excluded {n_before - len(df):,})")

# ─── STEP 7: EXCLUDE AUTOPSY / DCO CASES ─────────────────────────────────────

log("\n--- Step 7: Exclude autopsy and death certificate only (DCO) cases ---")
n_before = len(df)
if "sequence_number" in df.columns:
    df["sequence_number"] = df["sequence_number"].astype(str).str.strip()
    # SEER codes 60+ = DCO; code 0 = only tumor
    df = df[~df["sequence_number"].isin(["60", "61", "62", "99"])]
    log(f"  After DCO exclusion:{len(df):,}  (excluded {n_before - len(df):,})")
else:
    log("  sequence_number not found — DCO exclusion skipped")

# ─── STEP 8: KNOWN RACE/ETHNICITY ────────────────────────────────────────────

log("\n--- Step 8: Exclude unknown race/ethnicity ---")
n_before = len(df)
if "race_eth" in df.columns:
    df = df[~df["race_eth"].astype(str).isin(["Unknown", "nan", ""])]
log(f"  Known race/eth:     {len(df):,}  (excluded {n_before - len(df):,})")

# ─── VARIABLE ENGINEERING ────────────────────────────────────────────────────

log("\n--- Engineering analysis variables ---")

# Age groups
df["age_group"] = pd.cut(df["age_dx"],
                          bins=[0, 54, 64, 74, 999],
                          labels=["<55", "55-64", "65-74", "75+"])

# ICI era (exclude 2016 as transition year)
df["ici_era"] = np.where(df["year_dx"] <= PRE_ICI_END, "Pre-ICI (2004-2015)",
               np.where(df["year_dx"] >= POST_ICI_START, "Post-ICI (2017-2021)",
               "Transition (2016)"))

# Binary ICI era flag (excludes 2016)
df["post_ici"] = np.where(df["year_dx"] <= PRE_ICI_END, 0,
                np.where(df["year_dx"] >= POST_ICI_START, 1, np.nan))

# Rural-urban classification
if "rucc" in df.columns:
    df["rucc_num"] = pd.to_numeric(df["rucc"], errors="coerce")
    df["rurality"] = np.where(df["rucc_num"].between(1, 3), "Metropolitan",
                    np.where(df["rucc_num"].between(4, 6), "Urban",
                    np.where(df["rucc_num"].between(7, 9), "Rural", "Unknown")))

# Surgery receipt flag
if "surgery_code" in df.columns:
    df["surgery_code_num"] = pd.to_numeric(df["surgery_code"], errors="coerce")
    # SEER surgery codes: 0=none, 1-80=surgery types, 99=unknown
    df["surgery_received"] = np.where(df["surgery_code_num"].between(1, 80), 1,
                             np.where(df["surgery_code_num"] == 0, 0, np.nan))

    # Radical cystectomy specifically (SEER bladder surgery codes 50-75 typically)
    df["radical_cystectomy"] = np.where(df["surgery_code_num"].between(50, 75), 1, 0)

# Radiation receipt flag
if "radiation" in df.columns:
    df["radiation"] = df["radiation"].astype(str).str.strip()
    df["radiation_received"] = np.where(
        df["radiation"].isin(["Yes", "1", "Beam radiation", "Combination",
                              "Radioactive implants", "Radioisotopes",
                              "Refused (1988+)", "Recommended, unknown if administered"]),
        1,
        np.where(df["radiation"].isin(["None/Unknown", "0", "Unknown"]), 0, np.nan)
    )

# No treatment flag (primary market access outcome)
if "surgery_received" in df.columns and "radiation_received" in df.columns:
    df["no_treatment"] = np.where(
        (df["surgery_received"] == 0) & (df["radiation_received"] == 0), 1,
        np.where(
            (df["surgery_received"].isna()) & (df["radiation_received"].isna()), np.nan,
            0
        )
    )
    log(f"  No treatment rate:  {df['no_treatment'].mean():.1%}")

# Vital status binary
if "vital_status" in df.columns:
    df["vital_status"] = df["vital_status"].astype(str).str.strip()
    df["dead"] = np.where(df["vital_status"].isin(["Dead", "1"]), 1, 0)

# Cancer-specific death flag
if "css_death" in df.columns:
    df["css_death"] = df["css_death"].astype(str).str.strip()
    df["cancer_death"] = np.where(
        df["css_death"].str.contains("Bladder|bladder|cancer|Cancer", na=False), 1,
        np.where(df["dead"] == 1, 0, 0)
    )

# Stage simplification
if "ajcc_stage" in df.columns:
    df["stage_simple"] = df["ajcc_stage"].replace({
        "IIA": "II", "IIB": "II", "IIC": "II",
        "IIIA": "III", "IIIB": "III",
        "IVA": "IV", "IVB": "IV"
    })

# Survival months numeric
df["survival_months"] = pd.to_numeric(df["survival_months"], errors="coerce")
df["survival_years"]  = df["survival_months"] / 12

# Grade simplification
if "grade" in df.columns:
    df["grade"] = df["grade"].astype(str).str.strip()
    df["high_grade"] = np.where(
        df["grade"].isin(["Poorly differentiated", "Undifferentiated", "3", "4",
                          "High Grade", "High grade"]), 1,
        np.where(df["grade"].isin(["Well differentiated", "Moderately differentiated",
                                    "1", "2", "Low Grade", "Low grade"]), 0, np.nan)
    )

# Marital status binary
if "marital_status" in df.columns:
    df["married"] = np.where(
        df["marital_status"].astype(str).str.contains("Married|married", na=False), 1,
        np.where(df["marital_status"].astype(str).isin(["Unknown", "nan"]), np.nan, 0)
    )

log(f"\nFinal analytic cohort:   {len(df):,} patients")
log(f"  Pre-ICI (2004-2015): {(df['ici_era'] == 'Pre-ICI (2004-2015)').sum():,}")
log(f"  Transition (2016):   {(df['ici_era'] == 'Transition (2016)').sum():,}")
log(f"  Post-ICI (2017-2021):{(df['ici_era'] == 'Post-ICI (2017-2021)').sum():,}")

# ─── SAVE ────────────────────────────────────────────────────────────────────

os.makedirs("outputs", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
log(f"\nCleaned cohort saved to: {OUTPUT_PATH}")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))
log(f"Build log saved to:      {LOG_PATH}")
log("\nNext step: run 02_descriptive_table1.py")
