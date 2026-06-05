"""
Paper 1 - Script 5: Predictors of No Treatment Receipt (Primary Market Access Outcome)
========================================================================================
Produces:
  - Table 3: Multivariable logistic regression (AOR, 95% CI)
  - Figure 6: No-treatment rate heatmap by race x rurality x era
  - outputs/table3_no_treatment_logistic.csv
  - outputs/figures/figure6_no_treatment_heatmap.png

This is the KEY market access research question:
"Which patients receive NO cancer-directed treatment, and why?"

Input:  outputs/cleaned_cohort.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import os

INPUT_PATH = "outputs/cleaned_cohort.csv"
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv(INPUT_PATH, low_memory=False)
df = df[df["ici_era"] != "Transition (2016)"].copy()

if "no_treatment" not in df.columns:
    print("ERROR: no_treatment column not found. Ensure 01_cohort_identification.py ran successfully.")
    exit(1)

# Restrict to patients with known treatment status
df_lr = df[df["no_treatment"].notna()].copy()
print(f"Logistic regression cohort: N={len(df_lr):,}")
print(f"No treatment proportion: {df_lr['no_treatment'].mean():.1%}")

# ─── PREPARE VARIABLES ───────────────────────────────────────────────────────

# Reference categories (first alphabetically / clinically appropriate)
df_lr["age_group"]   = df_lr["age_group"].astype(str)
df_lr["race_eth"]    = df_lr["race_eth"].astype(str)
df_lr["rurality"]    = df_lr["rurality"].astype(str)
df_lr["stage_simple"]= df_lr.get("stage_simple", "Unknown").astype(str)
df_lr["post_ici"]    = pd.to_numeric(df_lr["post_ici"], errors="coerce")
df_lr["age_dx"]      = pd.to_numeric(df_lr["age_dx"], errors="coerce")
df_lr["high_grade"]  = pd.to_numeric(df_lr.get("high_grade", 0), errors="coerce")
df_lr["married"]     = pd.to_numeric(df_lr.get("married", np.nan), errors="coerce")

# ─── LOGISTIC REGRESSION ─────────────────────────────────────────────────────

print("\nFitting logistic regression models...")

def logistic_table(result, model_name):
    """Extract AOR, 95% CI, p-value from statsmodels logit result."""
    rows = []
    for var in result.params.index:
        if var == "Intercept":
            continue
        coef  = result.params[var]
        se    = result.bse[var]
        aor   = np.exp(coef)
        lower = np.exp(coef - 1.96 * se)
        upper = np.exp(coef + 1.96 * se)
        p     = result.pvalues[var]
        rows.append({
            "model":     model_name,
            "variable":  var,
            "AOR":       round(aor, 3),
            "CI_lower":  round(lower, 3),
            "CI_upper":  round(upper, 3),
            "p_value":   round(p, 4),
            "sig":       "*" if p < 0.05 else ""
        })
    return pd.DataFrame(rows)

all_tables = []

# Model 1: Unadjusted — ICI era
try:
    m1 = smf.logit("no_treatment ~ post_ici", data=df_lr).fit(disp=0)
    all_tables.append(logistic_table(m1, "Model 1: Unadjusted"))
    print(f"  Model 1 AUC: {m1.prsquared:.3f} (pseudo R²)")
except Exception as e:
    print(f"  Model 1 failed: {e}")

# Model 2: Clinical adjustment
try:
    m2 = smf.logit(
        "no_treatment ~ post_ici + age_dx + C(stage_simple, Treatment('II'))"
        " + high_grade + married",
        data=df_lr.dropna(subset=["age_dx", "stage_simple", "high_grade", "married"])
    ).fit(disp=0)
    all_tables.append(logistic_table(m2, "Model 2: Clinical"))
    print(f"  Model 2 AUC: {m2.prsquared:.3f} (pseudo R²)")
except Exception as e:
    print(f"  Model 2 failed: {e}")

# Model 3: Fully adjusted (primary model for publication)
try:
    df_m3 = df_lr.dropna(subset=["age_dx", "stage_simple", "high_grade",
                                   "married", "rurality", "post_ici"])
    m3 = smf.logit(
        "no_treatment ~ post_ici + age_dx"
        " + C(stage_simple, Treatment('II'))"
        " + high_grade + married"
        " + C(race_eth, Treatment('Non-Hispanic White'))"
        " + C(rurality, Treatment('Metropolitan'))",
        data=df_m3
    ).fit(disp=0)
    all_tables.append(logistic_table(m3, "Model 3: Fully Adjusted"))
    print(f"  Model 3 AUC: {m3.prsquared:.3f} (pseudo R²)")
    print(f"  Model 3 N:   {int(m3.nobs):,}")
except Exception as e:
    print(f"  Model 3 failed: {e}")

if all_tables:
    table3 = pd.concat(all_tables, ignore_index=True)
    table3.to_csv("outputs/table3_no_treatment_logistic.csv", index=False)
    print(f"\nTable 3 saved: outputs/table3_no_treatment_logistic.csv")

    # Print formatted Model 3 results
    m3_rows = table3[table3["model"] == "Model 3: Fully Adjusted"].copy()
    if not m3_rows.empty:
        print("\n  Model 3 (Fully Adjusted) — Key Results:")
        print(f"  {'Variable':<45} {'AOR':>6}  {'95% CI':<15}  {'p':>7}  {'Sig'}")
        print("  " + "-" * 80)
        for _, r in m3_rows.iterrows():
            ci = f"({r['CI_lower']:.2f}, {r['CI_upper']:.2f})"
            print(f"  {r['variable']:<45} {r['AOR']:>6.2f}  {ci:<15}  {r['p_value']:>7.4f}  {r['sig']}")

# ─── NO-TREATMENT RATE BY RACE × RURALITY × ERA ──────────────────────────────

print("\nGenerating no-treatment heatmap...")

if "race_eth" in df.columns and "rurality" in df.columns:
    eras = {"Pre-ICI": df[df["ici_era"] == "Pre-ICI (2004-2015)"],
            "Post-ICI": df[df["ici_era"] == "Post-ICI (2017-2021)"]}

    races = [r for r in ["Non-Hispanic White", "Non-Hispanic Black",
                          "Hispanic", "Asian/Pacific Islander"]
             if r in df["race_eth"].values]
    rurals = ["Metropolitan", "Urban", "Rural"]

    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
    fig6.suptitle("Proportion Receiving No Cancer-Directed Treatment\n"
                  "by Race/Ethnicity and Rurality",
                  fontsize=12, fontweight="bold")

    for ax, (era_label, era_df) in zip(axes6, eras.items()):
        matrix = np.full((len(races), len(rurals)), np.nan)
        annot  = np.empty((len(races), len(rurals)), dtype=object)

        for i, race in enumerate(races):
            for j, rural in enumerate(rurals):
                sub = era_df[(era_df["race_eth"] == race) &
                             (era_df["rurality"] == rural) &
                             (era_df["no_treatment"].notna())]
                if len(sub) >= 20:
                    rate = sub["no_treatment"].mean() * 100
                    matrix[i, j] = rate
                    annot[i, j]  = f"{rate:.0f}%\n(n={len(sub):,})"
                else:
                    annot[i, j] = "n<20"

        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=80, aspect="auto")
        ax.set_xticks(range(len(rurals)))
        ax.set_yticks(range(len(races)))
        ax.set_xticklabels(rurals, fontsize=9)
        ax.set_yticklabels(races, fontsize=9)
        ax.set_title(f"{era_label}", fontsize=11, fontweight="bold")

        for i in range(len(races)):
            for j in range(len(rurals)):
                val = matrix[i, j]
                color = "white" if (not np.isnan(val) and (val > 50 or val < 20)) else "black"
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="% No treatment", shrink=0.85)

    plt.tight_layout()
    plt.savefig("outputs/figures/figure6_no_treatment_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 6 saved: outputs/figures/figure6_no_treatment_heatmap.png")

# ─── SUBGROUP NO-TREATMENT RATES ─────────────────────────────────────────────

print("\nNo-treatment rates by key subgroups:")
subgroup_rates = []

for col, groups in [
    ("ici_era",      ["Pre-ICI (2004-2015)", "Post-ICI (2017-2021)"]),
    ("race_eth",     df["race_eth"].dropna().value_counts().index[:5].tolist()),
    ("rurality",     ["Metropolitan", "Urban", "Rural"]),
    ("stage_simple", ["II", "III", "IV"]),
    ("age_group",    ["<55", "55-64", "65-74", "75+"]),
]:
    if col not in df.columns:
        continue
    for grp in groups:
        sub = df[(df[col].astype(str) == str(grp)) & df["no_treatment"].notna()]
        if len(sub) < 20:
            continue
        rate = sub["no_treatment"].mean() * 100
        subgroup_rates.append({"subgroup": col, "group": grp,
                                "n": len(sub), "pct_no_treatment": round(rate, 1)})
        print(f"  {col} = {grp:<30} n={len(sub):>6,}  no-treatment: {rate:.1f}%")

pd.DataFrame(subgroup_rates).to_csv("outputs/subgroup_no_treatment.csv", index=False)

print("\nAll no-treatment analyses complete.")
print("Next step: run 06_sensitivity_analyses.py")
