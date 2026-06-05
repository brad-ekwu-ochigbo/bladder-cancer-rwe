"""
Paper 1 - Script 4: Survival Analysis
======================================
Produces:
  - Kaplan-Meier curves (OS and CSS) by key subgroups
  - Fine-Gray competing risk analysis (cancer vs. other-cause death)
  - Multivariable Cox proportional hazards models (3 models)
  - outputs/km_results.csv
  - outputs/cox_results.csv
  - outputs/competing_risk_results.csv
  - outputs/figures/figure3_km_curves.png
  - outputs/figures/figure4_competing_risk.png
  - outputs/figures/figure5_forest_plot.png

Input:  outputs/cleaned_cohort.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import AalenJohansenFitter
import warnings
warnings.filterwarnings("ignore")
import os

INPUT_PATH = "outputs/cleaned_cohort.csv"
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv(INPUT_PATH, low_memory=False)
df = df[df["ici_era"] != "Transition (2016)"].copy()

# Ensure required columns
df["survival_months"] = pd.to_numeric(df["survival_months"], errors="coerce")
df["dead"]            = pd.to_numeric(df.get("dead", 0), errors="coerce").fillna(0)
df["cancer_death"]    = pd.to_numeric(df.get("cancer_death", 0), errors="coerce").fillna(0)

# Remove rows with missing survival
df = df[df["survival_months"].notna() & (df["survival_months"] >= 0)].copy()
print(f"Analytic cohort for survival: N={len(df):,}")

# Event indicator for competing risk
# 0 = censored, 1 = cancer death, 2 = other-cause death
if "cancer_death" in df.columns and "dead" in df.columns:
    df["event_cr"] = np.where(df["cancer_death"] == 1, 1,
                    np.where((df["dead"] == 1) & (df["cancer_death"] == 0), 2, 0))

# ─── KAPLAN-MEIER ────────────────────────────────────────────────────────────

def km_summary(col, groups, duration="survival_months", event="dead",
               label="Group"):
    """Return KM median survival and log-rank p-value for multiple groups."""
    results = []
    kmf = KaplanMeierFitter()
    for grp in groups:
        subset = df[df[col].astype(str) == str(grp)]
        if len(subset) < 20:
            continue
        kmf.fit(subset[duration], event_observed=subset[event],
                label=str(grp))
        median = kmf.median_survival_time_
        ci_low, ci_hi = (kmf.confidence_interval_.iloc[
            kmf.confidence_interval_.index.get_loc(
                kmf.confidence_interval_.index[
                    np.searchsorted(kmf.survival_function_.values.flatten(),
                                   0.5, side="right")
                ]
            )
        ].values if len(kmf.confidence_interval_) > 0 else (np.nan, np.nan))
        results.append({"group": grp, "n": len(subset), "median_os": round(median, 1)})

    # Log-rank test
    try:
        grp_data = [df[df[col].astype(str) == str(g)][duration] for g in groups]
        grp_evts = [df[df[col].astype(str) == str(g)][event] for g in groups]
        if len(grp_data) >= 2:
            result = multivariate_logrank_test(
                pd.concat(grp_data),
                pd.concat([pd.Series([str(g)] * len(d))
                           for g, d in zip(groups, grp_data)]),
                event_observed=pd.concat(grp_evts)
            )
            for r in results:
                r["logrank_p"] = round(result.p_value, 4)
    except Exception:
        pass
    return results

print("\nRunning Kaplan-Meier analyses...")

km_rows = []
# By ICI era
if "ici_era" in df.columns:
    eras = ["Pre-ICI (2004-2015)", "Post-ICI (2017-2021)"]
    res = km_summary("ici_era", eras)
    for r in res: r["stratifier"] = "ICI era"; km_rows.append(r)

# By race/ethnicity
if "race_eth" in df.columns:
    races = df["race_eth"].dropna().value_counts().index[:5].tolist()
    res = km_summary("race_eth", races)
    for r in res: r["stratifier"] = "Race/ethnicity"; km_rows.append(r)

# By rurality
if "rurality" in df.columns:
    res = km_summary("rurality", ["Metropolitan", "Urban", "Rural"])
    for r in res: r["stratifier"] = "Rurality"; km_rows.append(r)

# By stage
if "stage_simple" in df.columns:
    res = km_summary("stage_simple", ["II", "III", "IV"])
    for r in res: r["stratifier"] = "AJCC Stage"; km_rows.append(r)

# By no_treatment
if "no_treatment" in df.columns:
    res = km_summary("no_treatment", [0, 1])
    for r in res: r["stratifier"] = "No treatment"; km_rows.append(r)

pd.DataFrame(km_rows).to_csv("outputs/km_results.csv", index=False)
print(f"  KM results saved: {len(km_rows)} rows")

# ─── FIGURE 3: KM CURVES ─────────────────────────────────────────────────────

PALETTE = ["#2E75B6", "#C00000", "#548235", "#7030A0", "#ED7D31"]

def plot_km(ax, col, groups, title, duration="survival_months", event="dead"):
    kmf = KaplanMeierFitter()
    for i, grp in enumerate(groups):
        subset = df[df[col].astype(str) == str(grp)]
        if len(subset) < 20:
            continue
        kmf.fit(subset[duration], event_observed=subset[event], label=str(grp))
        kmf.plot_survival_function(ax=ax, color=PALETTE[i % len(PALETTE)],
                                   ci_show=True, ci_alpha=0.1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Time (months)", fontsize=9)
    ax.set_ylabel("Survival probability", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.flatten()
fig3.suptitle("Overall Survival by Patient and Clinical Subgroups\nBladder Cancer SEER Analysis, 2004-2021",
              fontsize=13, fontweight="bold", y=1.01)

if "ici_era" in df.columns:
    plot_km(axes3[0], "ici_era",
            ["Pre-ICI (2004-2015)", "Post-ICI (2017-2021)"],
            "OS by ICI era")

if "race_eth" in df.columns:
    races = df["race_eth"].dropna().value_counts().index[:4].tolist()
    plot_km(axes3[1], "race_eth", races, "OS by Race/Ethnicity")

if "rurality" in df.columns:
    plot_km(axes3[2], "rurality", ["Metropolitan", "Urban", "Rural"],
            "OS by Rural-Urban Classification")

if "stage_simple" in df.columns:
    plot_km(axes3[3], "stage_simple", ["II", "III", "IV"],
            "OS by AJCC Stage")

plt.tight_layout()
plt.savefig("outputs/figures/figure3_km_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 3 (KM curves) saved.")

# ─── COMPETING RISK (FINE-GRAY) ──────────────────────────────────────────────

print("\nRunning competing risk analysis (cancer vs. other-cause death)...")

if "event_cr" in df.columns:
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle("Cumulative Incidence Functions: Cancer-Specific vs. Other-Cause Death\nFine-Gray Competing Risk Analysis",
                  fontsize=12, fontweight="bold")

    for ax, (col, groups, title) in zip(axes4, [
        ("ici_era",  ["Pre-ICI (2004-2015)", "Post-ICI (2017-2021)"], "By ICI era"),
        ("rurality", ["Metropolitan", "Urban", "Rural"], "By Rurality")
    ]):
        ajf = AalenJohansenFitter(calculate_variance=True)
        for i, grp in enumerate(groups):
            sub = df[df[col].astype(str) == str(grp)]
            if len(sub) < 20:
                continue
            try:
                ajf.fit(sub["survival_months"], sub["event_cr"],
                        event_of_interest=1, label=f"{grp} (cancer)")
                ajf.plot(ax=ax, color=PALETTE[i], linestyle="-")
                ajf.fit(sub["survival_months"], sub["event_cr"],
                        event_of_interest=2, label=f"{grp} (other)")
                ajf.plot(ax=ax, color=PALETTE[i], linestyle="--", alpha=0.6)
            except Exception as e:
                print(f"  Competing risk failed for {grp}: {e}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (months)", fontsize=9)
        ax.set_ylabel("Cumulative incidence", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/figures/figure4_competing_risk.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 4 (competing risk CIF) saved.")

# ─── COX PROPORTIONAL HAZARDS ────────────────────────────────────────────────

print("\nRunning Cox proportional hazards models...")

# Prepare Cox dataset — dummy encode categorical variables
cox_df = df[["survival_months", "dead", "post_ici",
             "age_dx", "stage_simple", "high_grade", "married",
             "surgery_received", "radiation_received", "no_treatment",
             "race_eth", "rurality", "income_quartile"]].copy()

cox_df = cox_df[cox_df["survival_months"] > 0].copy()

# Dummy encode
cat_cols = ["stage_simple", "race_eth", "rurality", "income_quartile"]
cox_df = pd.get_dummies(cox_df, columns=cat_cols, drop_first=True, dtype=float)

# Fill numeric NAs with median
for col in cox_df.select_dtypes(include=[np.number]).columns:
    cox_df[col] = cox_df[col].fillna(cox_df[col].median())

cox_results = []

def run_cox(name, covariates, df_cox):
    """Run Cox model and return tidy results."""
    avail = [c for c in covariates if c in df_cox.columns]
    if not avail:
        print(f"  {name}: no valid covariates found — skipping")
        return
    try:
        cph = CoxPHFitter()
        cph.fit(df_cox[avail + ["survival_months", "dead"]],
                duration_col="survival_months",
                event_col="dead",
                show_progress=False)
        summary = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                                "exp(coef) upper 95%", "p"]].copy()
        summary.columns = ["coef", "HR", "HR_lower", "HR_upper", "p_value"]
        summary["model"] = name
        summary["covariate"] = summary.index
        summary = summary.reset_index(drop=True)
        cox_results.append(summary)
        print(f"  {name}: concordance = {cph.concordance_index_:.3f}")
        return cph
    except Exception as e:
        print(f"  {name}: failed — {e}")
        return None

# Model 1: Unadjusted (ICI era only)
run_cox("Model 1 (unadjusted)", ["post_ici"], cox_df)

# Model 2: Age, sex, stage, grade, marital status
m2_vars = ["post_ici", "age_dx", "high_grade", "married",
           "surgery_received", "radiation_received"] + \
          [c for c in cox_df.columns if c.startswith("stage_simple_")]
run_cox("Model 2 (clinical)", m2_vars, cox_df)

# Model 3: Fully adjusted
m3_vars = m2_vars + \
          [c for c in cox_df.columns if c.startswith("race_eth_")] + \
          [c for c in cox_df.columns if c.startswith("rurality_")] + \
          [c for c in cox_df.columns if c.startswith("income_quartile_")]
cph3 = run_cox("Model 3 (fully adjusted)", m3_vars, cox_df)

if cox_results:
    all_cox = pd.concat(cox_results, ignore_index=True)
    all_cox = all_cox[["model", "covariate", "coef", "HR",
                        "HR_lower", "HR_upper", "p_value"]]
    all_cox = all_cox.round(3)
    all_cox.to_csv("outputs/cox_results.csv", index=False)
    print(f"Cox results saved: outputs/cox_results.csv")

# ─── FIGURE 5: FOREST PLOT ───────────────────────────────────────────────────

if cph3 is not None:
    print("\nGenerating forest plot...")
    try:
        summary = cph3.summary.copy()
        summary["HR"]    = np.exp(summary["coef"])
        summary["lower"] = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
        summary["upper"] = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])
        summary["sig"]   = summary["p"] < 0.05

        # Select key covariates for forest plot
        plot_vars = [c for c in summary.index
                     if any(k in c for k in
                            ["post_ici", "age", "stage", "race", "rural",
                             "income", "no_treatment", "surgery", "radiation"])]
        plot_df = summary.loc[plot_vars].copy() if plot_vars else summary.head(20)

        fig5, ax5 = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))
        y_pos = range(len(plot_df))

        colors_fp = ["#C00000" if s else "#2E75B6"
                     for s in plot_df["sig"]]
        ax5.hlines(y_pos, plot_df["lower"], plot_df["upper"],
                   color=colors_fp, linewidth=2.5, alpha=0.7)
        ax5.scatter(plot_df["HR"], y_pos,
                    color=colors_fp, s=80, zorder=5)
        ax5.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.6)

        ax5.set_yticks(list(y_pos))
        ax5.set_yticklabels(plot_df.index, fontsize=9)
        ax5.set_xlabel("Hazard Ratio (95% CI)", fontsize=10)
        ax5.set_title("Forest Plot: Multivariable Cox Regression\n"
                      "Overall Survival — Fully Adjusted Model",
                      fontsize=11, fontweight="bold")
        ax5.grid(True, axis="x", alpha=0.3)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#C00000", label="p < 0.05"),
                           Patch(facecolor="#2E75B6", label="p \u2265 0.05")]
        ax5.legend(handles=legend_elements, fontsize=9, loc="lower right")

        plt.tight_layout()
        plt.savefig("outputs/figures/figure5_forest_plot.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("Figure 5 (forest plot) saved.")
    except Exception as e:
        print(f"  Forest plot failed: {e}")

print("\nAll survival analyses complete.")
print("Next step: run 05_logistic_no_treatment.py")
