"""
Paper 1 - Script 2: Descriptive Statistics (Table 1)
=====================================================
Produces Table 1: Patient characteristics by ICI era (pre vs. post)
with chi-square tests and standardized mean differences (SMD).

Input:  outputs/cleaned_cohort.csv
Output: outputs/table1.csv
        outputs/table1_formatted.xlsx
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

INPUT_PATH  = "outputs/cleaned_cohort.csv"
OUTPUT_CSV  = "outputs/table1.csv"
OUTPUT_XLSX = "outputs/table1_formatted.xlsx"

df = pd.read_csv(INPUT_PATH, low_memory=False)
# Restrict to pre/post ICI only (exclude transition year 2016)
df = df[df["ici_era"] != "Transition (2016)"].copy()

pre  = df[df["ici_era"] == "Pre-ICI (2004-2015)"]
post = df[df["ici_era"] == "Post-ICI (2017-2021)"]

print(f"Pre-ICI n={len(pre):,}  |  Post-ICI n={len(post):,}")

rows = []

def smd_cat(col, val, g1, g2):
    """Standardized mean difference for binary/categorical."""
    p1 = (g1[col].astype(str) == str(val)).mean()
    p2 = (g2[col].astype(str) == str(val)).mean()
    pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
    return round((p1 - p2) / np.sqrt(pooled_var), 3) if pooled_var > 0 else 0

def smd_cont(col, g1, g2):
    """Standardized mean difference for continuous variables."""
    m1, s1 = g1[col].mean(), g1[col].std()
    m2, s2 = g2[col].mean(), g2[col].std()
    pooled_sd = np.sqrt((s1**2 + s2**2) / 2)
    return round((m1 - m2) / pooled_sd, 3) if pooled_sd > 0 else 0

def add_header(label):
    rows.append({"Variable": label, "Overall": "", "Pre-ICI": "", "Post-ICI": "",
                 "p-value": "", "SMD": ""})

def add_n():
    add_header("N (%)")
    rows.append({
        "Variable": "  Total",
        "Overall":  f"{len(df):,}",
        "Pre-ICI":  f"{len(pre):,} (100.0%)",
        "Post-ICI": f"{len(post):,} (100.0%)",
        "p-value": "",
        "SMD": ""
    })

def add_cat(label, col, vals, ref_val=None):
    """Add categorical variable rows with chi-square p-value."""
    add_header(label)
    cont_table = []
    for val in vals:
        n_all = (df[col].astype(str) == str(val)).sum()
        n_pre = (pre[col].astype(str) == str(val)).sum()
        n_pos = (post[col].astype(str) == str(val)).sum()
        p_pre = n_pre / len(pre) * 100 if len(pre) > 0 else 0
        p_pos = n_pos / len(post) * 100 if len(post) > 0 else 0
        p_all = n_all / len(df) * 100 if len(df) > 0 else 0
        smd_val = smd_cat(col, val, pre, post) if n_all > 5 else ""
        rows.append({
            "Variable": f"  {val}",
            "Overall":  f"{n_all:,} ({p_all:.1f}%)",
            "Pre-ICI":  f"{n_pre:,} ({p_pre:.1f}%)",
            "Post-ICI": f"{n_pos:,} ({p_pos:.1f}%)",
            "p-value":  "",
            "SMD":      smd_val
        })
        cont_table.append([n_pre, n_pos])
    # Chi-square on full contingency table
    try:
        chi2, p, _, _ = stats.chi2_contingency(cont_table)
        rows[-len(vals)]["p-value"] = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    except Exception:
        rows[-len(vals)]["p-value"] = "N/A"

def add_cont(label, col):
    """Add continuous variable row with mean (SD), median (IQR), t-test."""
    pre_vals  = pre[col].dropna()
    post_vals = post[col].dropna()
    all_vals  = df[col].dropna()

    def fmt(s): return (f"{s.mean():.1f} ({s.std():.1f}) | "
                        f"median {s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]")

    try:
        _, p = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
        p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    except Exception:
        p_str = "N/A"

    rows.append({
        "Variable": label + "  mean (SD) | median [IQR]",
        "Overall":  fmt(all_vals),
        "Pre-ICI":  fmt(pre_vals),
        "Post-ICI": fmt(post_vals),
        "p-value":  p_str,
        "SMD":      smd_cont(col, pre, post)
    })

# ─── BUILD TABLE ─────────────────────────────────────────────────────────────

add_n()

# Age
if "age_dx" in df.columns:
    add_cont("Age at diagnosis (years)", "age_dx")
if "age_group" in df.columns:
    add_cat("Age group", "age_group", ["<55", "55-64", "65-74", "75+"])

# Sex
if "sex" in df.columns:
    sex_vals = df["sex"].dropna().unique().tolist()
    add_cat("Sex", "sex", sorted(sex_vals))

# Race/ethnicity
if "race_eth" in df.columns:
    race_vals = df["race_eth"].dropna().value_counts().index.tolist()
    add_cat("Race/ethnicity", "race_eth", race_vals)

# Rurality
if "rurality" in df.columns:
    add_cat("Rural-urban classification", "rurality",
            ["Metropolitan", "Urban", "Rural", "Unknown"])

# Income
if "income_quartile" in df.columns:
    inc_vals = sorted(df["income_quartile"].dropna().unique().tolist())
    add_cat("County income quartile", "income_quartile", inc_vals)

# Marital status
if "marital_status" in df.columns:
    mar_vals = df["marital_status"].dropna().value_counts().index.tolist()
    add_cat("Marital status", "marital_status", mar_vals[:5])

# Stage
if "stage_simple" in df.columns:
    add_cat("AJCC Stage", "stage_simple", ["II", "III", "IV"])
elif "ajcc_stage" in df.columns:
    stage_vals = df["ajcc_stage"].dropna().value_counts().index.tolist()
    add_cat("AJCC Stage", "ajcc_stage", stage_vals[:6])

# Grade
if "high_grade" in df.columns:
    add_header("Tumor grade")
    for era_label, grp in [("", df), ("Pre-ICI", pre), ("Post-ICI", post)]:
        pass
    hg_all = (df["high_grade"] == 1).sum()
    hg_pre = (pre["high_grade"] == 1).sum()
    hg_pos = (post["high_grade"] == 1).sum()
    rows.append({
        "Variable": "  High grade",
        "Overall":  f"{hg_all:,} ({hg_all/len(df)*100:.1f}%)",
        "Pre-ICI":  f"{hg_pre:,} ({hg_pre/len(pre)*100:.1f}%)",
        "Post-ICI": f"{hg_pos:,} ({hg_pos/len(post)*100:.1f}%)",
        "p-value":  "",
        "SMD":      smd_cat("high_grade", 1, pre, post)
    })

# Treatment receipt
add_header("Treatment receipt")
for (label, col) in [("Surgery received", "surgery_received"),
                      ("Radiation received", "radiation_received"),
                      ("No cancer-directed treatment", "no_treatment"),
                      ("Radical cystectomy", "radical_cystectomy")]:
    if col in df.columns:
        n_all = (df[col] == 1).sum()
        n_pre = (pre[col] == 1).sum()
        n_pos = (post[col] == 1).sum()
        try:
            chi2, p, _, _ = stats.chi2_contingency([
                [n_pre, len(pre) - n_pre],
                [n_pos, len(post) - n_pos]
            ])
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        except Exception:
            p_str = "N/A"
        rows.append({
            "Variable": f"  {label}",
            "Overall":  f"{n_all:,} ({n_all/len(df)*100:.1f}%)",
            "Pre-ICI":  f"{n_pre:,} ({n_pre/len(pre)*100:.1f}%)",
            "Post-ICI": f"{n_pos:,} ({n_pos/len(post)*100:.1f}%)",
            "p-value":  p_str,
            "SMD":      smd_cat(col, 1, pre, post)
        })

# Survival
if "survival_months" in df.columns:
    add_cont("Survival (months)", "survival_months")

# ─── SAVE ────────────────────────────────────────────────────────────────────

table1 = pd.DataFrame(rows)
table1.to_csv(OUTPUT_CSV, index=False)
print(f"\nTable 1 saved to: {OUTPUT_CSV}")

# Formatted Excel
try:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        table1.to_excel(writer, index=False, sheet_name="Table 1")
        ws = writer.sheets["Table 1"]
        ws.column_dimensions["A"].width = 42
        for col in ["B", "C", "D", "E", "F"]:
            ws.column_dimensions[col].width = 22
    print(f"Formatted Excel saved to: {OUTPUT_XLSX}")
except ImportError:
    print("openpyxl not installed — Excel output skipped. CSV saved.")

print("\nNext step: run 03_trend_analysis.py")
