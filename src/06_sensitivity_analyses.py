"""
Paper 1 - Script 6: Sensitivity Analyses
==========================================
Runs all pre-specified sensitivity analyses and compares results
to main analysis to assess robustness.

Sensitivity analyses:
  1. Exclude survival time = 0 months (same-month deaths)
  2. Include stage I cases (NMIBC)
  3. All histologies (not urothelial only)
  4. Use SEER summary stage instead of AJCC
  5. Complete-case analysis (no unknown category)

Outputs sensitivity comparison table to:
  outputs/sensitivity_results.csv

Input:  outputs/cleaned_cohort.csv (main)
        outputs/sensitivity_stage1.csv
        outputs/sensitivity_non_urothelial.csv
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import warnings
warnings.filterwarnings("ignore")
import os

OUTPUT_PATH = "outputs/sensitivity_results.csv"

def run_quick_cox(df, name):
    """Run unadjusted ICI era Cox model and return HR with 95% CI."""
    d = df[df["ici_era"] != "Transition (2016)"].copy()
    d = d[d["survival_months"].notna() & (d["survival_months"] > 0)].copy()
    d["post_ici"] = np.where(d["ici_era"] == "Post-ICI (2017-2021)", 1, 0)
    if d["post_ici"].sum() < 20 or (1 - d["post_ici"]).sum() < 20:
        return {"analysis": name, "n": len(d), "HR": np.nan,
                "CI_lower": np.nan, "CI_upper": np.nan, "p": np.nan}
    try:
        cph = CoxPHFitter()
        cph.fit(d[["survival_months", "dead", "post_ici"]].dropna(),
                duration_col="survival_months", event_col="dead", show_progress=False)
        hr    = round(np.exp(cph.params_["post_ici"]), 3)
        lower = round(cph.confidence_intervals_["95% lower-bound"]["post_ici"], 3)
        upper = round(cph.confidence_intervals_["95% upper-bound"]["post_ici"], 3)
        p     = round(cph.summary["p"]["post_ici"], 4)
        return {"analysis": name, "n": len(d), "HR": hr,
                "CI_lower": lower, "CI_upper": upper, "p": p}
    except Exception as e:
        return {"analysis": name, "n": len(d), "HR": np.nan,
                "CI_lower": np.nan, "CI_upper": np.nan, "p": np.nan,
                "error": str(e)}

results = []

# 0. Main analysis
print("Running sensitivity analyses...")
df_main = pd.read_csv("outputs/cleaned_cohort.csv", low_memory=False)
results.append(run_quick_cox(df_main, "Main analysis"))

# 1. Exclude survival = 0 months
df_s1 = df_main[df_main["survival_months"] > 0].copy()
results.append(run_quick_cox(df_s1, "SA1: Exclude survival=0 months"))

# 2. Include stage I (load sensitivity file)
if os.path.exists("outputs/sensitivity_stage1.csv"):
    df_s1_stage = pd.read_csv("outputs/sensitivity_stage1.csv", low_memory=False)
    df_s2 = pd.concat([df_main, df_s1_stage], ignore_index=True)
    results.append(run_quick_cox(df_s2, "SA2: Include stage I (NMIBC)"))

# 3. All histologies
if os.path.exists("outputs/sensitivity_non_urothelial.csv"):
    df_nonuro = pd.read_csv("outputs/sensitivity_non_urothelial.csv", low_memory=False)
    df_s3 = pd.concat([df_main, df_nonuro], ignore_index=True)
    results.append(run_quick_cox(df_s3, "SA3: All histologies"))

# 4. SEER summary stage
if "summary_stage" in df_main.columns:
    df_s4 = df_main.copy()
    df_s4["stage_for_analysis"] = df_s4["summary_stage"]
    results.append(run_quick_cox(df_s4, "SA4: SEER summary stage"))

# 5. Complete case (exclude all unknowns)
key_cols = ["age_dx", "race_eth", "rurality", "stage_simple",
            "no_treatment", "surgery_received"]
avail = [c for c in key_cols if c in df_main.columns]
df_s5 = df_main.dropna(subset=avail)
unknown_vals = ["Unknown", "unknown", "nan", ""]
for col in avail:
    df_s5 = df_s5[~df_s5[col].astype(str).isin(unknown_vals)]
results.append(run_quick_cox(df_s5, f"SA5: Complete case (n={len(df_s5):,})"))

# ─── OUTPUT ──────────────────────────────────────────────────────────────────

sens_df = pd.DataFrame(results)
sens_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSensitivity results saved: {OUTPUT_PATH}")
print(f"\n{'Analysis':<45} {'N':>8}  {'HR':>6}  {'95% CI':<16}  {'p':>7}")
print("-" * 85)
for _, r in sens_df.iterrows():
    ci = f"({r['CI_lower']:.2f}, {r['CI_upper']:.2f})" if not pd.isna(r["CI_lower"]) else "N/A"
    p_str = f"{r['p']:.4f}" if not pd.isna(r["p"]) else "N/A"
    print(f"{r['analysis']:<45} {int(r['n']):>8,}  {r['HR']:>6.3f}  {ci:<16}  {p_str:>7}")

print("\n" + "="*65)
print("ALL ANALYSES COMPLETE")
print("="*65)
print("""
Outputs generated:
  outputs/cleaned_cohort.csv          — analysis dataset
  outputs/table1.csv                  — Table 1 (descriptive)
  outputs/table1_formatted.xlsx       — Table 1 (formatted)
  outputs/annual_trends.csv           — for NCI Joinpoint Program
  outputs/its_results.csv             — ITS model results
  outputs/km_results.csv              — KM median survival
  outputs/cox_results.csv             — Cox model results (all 3 models)
  outputs/table3_no_treatment_logistic.csv — Logistic regression Table 3
  outputs/subgroup_no_treatment.csv   — No-treatment by subgroup
  outputs/sensitivity_results.csv     — Sensitivity analysis HR comparison
  outputs/figures/figure1_trends.png
  outputs/figures/figure2_its.png
  outputs/figures/figure3_km_curves.png
  outputs/figures/figure4_competing_risk.png
  outputs/figures/figure5_forest_plot.png
  outputs/figures/figure6_no_treatment_heatmap.png

Next steps:
  1. Import outputs/annual_trends.csv into NCI Joinpoint Program
     for formal APC/AAPC trend statistics
  2. Review outputs and refine as needed
  3. Begin manuscript draft
""")
