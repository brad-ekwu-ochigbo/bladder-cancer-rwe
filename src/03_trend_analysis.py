"""
Paper 1 - Script 3: Trend Analysis & Interrupted Time Series
=============================================================
Produces:
  - Annual trends in late-stage diagnosis and no-treatment rates
  - Interrupted time series (ITS) model for pre/post ICI era
  - Figure 1: Trend plots with joinpoint annotations
  - Figure 2: ITS segmented regression plots
  - outputs/annual_trends.csv
  - outputs/its_results.csv

Note: Full joinpoint regression requires the NCI Joinpoint Program
(free Windows download from surveillance.cancer.gov/joinpoint).
This script produces the annual rate data for that program AND
runs a Python segmented regression as an ITS proxy.

Input:  outputs/cleaned_cohort.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import os

INPUT_PATH = "outputs/cleaned_cohort.csv"
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv(INPUT_PATH, low_memory=False)

# Exclude transition year for ITS but include for trend
df_trend = df[df["year_dx"] != 2016].copy()

# ─── ANNUAL RATE CALCULATIONS ─────────────────────────────────────────────────

print("Computing annual rates...")
years = sorted(df["year_dx"].dropna().unique().astype(int))

annual = []
for yr in years:
    yr_df = df[df["year_dx"] == yr]
    n = len(yr_df)
    if n < 10:
        continue

    rec = {"year": yr, "n": n}

    # Late-stage rate (stage III or IV)
    if "stage_simple" in yr_df.columns:
        late = yr_df["stage_simple"].isin(["III", "IV"]).sum()
        rec["pct_late_stage"] = late / n * 100

    # No treatment rate
    if "no_treatment" in yr_df.columns:
        no_tx = (yr_df["no_treatment"] == 1).sum()
        rec["pct_no_treatment"] = no_tx / n * 100

    # Surgery rate
    if "surgery_received" in yr_df.columns:
        surg = (yr_df["surgery_received"] == 1).sum()
        rec["pct_surgery"] = surg / n * 100

    # Radiation rate
    if "radiation_received" in yr_df.columns:
        rad = (yr_df["radiation_received"] == 1).sum()
        rec["pct_radiation"] = rad / n * 100

    annual.append(rec)

annual_df = pd.DataFrame(annual)
annual_df.to_csv("outputs/annual_trends.csv", index=False)
print(f"Annual trends saved: {len(annual_df)} years of data")

# ─── INTERRUPTED TIME SERIES ──────────────────────────────────────────────────

def its_model(annual_df, outcome_col, ici_year=2016, post_start=2017):
    """
    Segmented regression ITS:
    Y = b0 + b1*time + b2*ici_era + b3*time_after_ici + e
    Returns fitted model and parameters.
    """
    d = annual_df[annual_df["year"] != ici_year].copy()
    d = d.dropna(subset=[outcome_col])
    d = d.sort_values("year")

    d["time"]          = d["year"] - d["year"].min()
    d["ici_era"]       = (d["year"] >= post_start).astype(int)
    d["time_after_ici"] = np.where(d["year"] >= post_start,
                                   d["year"] - post_start + 1, 0)

    X = sm.add_constant(d[["time", "ici_era", "time_after_ici"]])
    y = d[outcome_col]

    try:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        return model, d
    except Exception as e:
        print(f"  ITS model failed for {outcome_col}: {e}")
        return None, d

its_results = []
outcomes = ["pct_late_stage", "pct_no_treatment", "pct_surgery", "pct_radiation"]
outcome_labels = {
    "pct_late_stage":    "Late-stage diagnosis (stage III-IV)",
    "pct_no_treatment":  "No cancer-directed treatment",
    "pct_surgery":       "Surgical treatment receipt",
    "pct_radiation":     "Radiation therapy receipt"
}

print("\nRunning ITS models...")
for col in outcomes:
    if col not in annual_df.columns:
        continue
    model, d = its_model(annual_df, col)
    if model is None:
        continue
    params = model.params
    pvals  = model.pvalues
    cis    = model.conf_int()
    print(f"\n  {outcome_labels.get(col, col)}")
    print(f"    Intercept (baseline level):     {params.get('const', np.nan):.2f}")
    print(f"    Pre-ICI trend (beta1):          {params.get('time', np.nan):.3f}  p={pvals.get('time', np.nan):.4f}")
    print(f"    Level change at ICI (beta2):    {params.get('ici_era', np.nan):.2f}  p={pvals.get('ici_era', np.nan):.4f}")
    print(f"    Post-ICI slope change (beta3):  {params.get('time_after_ici', np.nan):.3f}  p={pvals.get('time_after_ici', np.nan):.4f}")
    its_results.append({
        "outcome":       outcome_labels.get(col, col),
        "intercept":     round(params.get("const", np.nan), 3),
        "beta_trend":    round(params.get("time", np.nan), 3),
        "p_trend":       round(pvals.get("time", np.nan), 4),
        "beta_level":    round(params.get("ici_era", np.nan), 3),
        "p_level":       round(pvals.get("ici_era", np.nan), 4),
        "beta_slope":    round(params.get("time_after_ici", np.nan), 3),
        "p_slope":       round(pvals.get("time_after_ici", np.nan), 4),
        "r_squared":     round(model.rsquared, 3),
        "n_years":       len(d)
    })

if its_results:
    pd.DataFrame(its_results).to_csv("outputs/its_results.csv", index=False)
    print("\nITS results saved to: outputs/its_results.csv")

# ─── FIGURE 1: TREND PLOTS ───────────────────────────────────────────────────

plot_outcomes = [(c, outcome_labels.get(c, c)) for c in outcomes if c in annual_df.columns]
if plot_outcomes:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Annual Trends in Bladder Cancer Treatment and Stage at Diagnosis\n"
        "SEER Population-Based Analysis, 2004-2021",
        fontsize=13, fontweight="bold", y=1.01
    )
    axes = axes.flatten()

    colors = {"pre": "#2E75B6", "post": "#C00000", "ici": "#888888"}

    for idx, (col, label) in enumerate(plot_outcomes):
        ax = axes[idx]
        d  = annual_df[["year", col]].dropna()

        # Pre and post series
        d_pre  = d[d["year"] <= 2015]
        d_post = d[d["year"] >= 2017]

        ax.plot(d_pre["year"],  d_pre[col],  "o-", color=colors["pre"],
                linewidth=2, markersize=5, label="Pre-ICI (2004-2015)")
        ax.plot(d_post["year"], d_post[col], "s-", color=colors["post"],
                linewidth=2, markersize=5, label="Post-ICI (2017-2021)")

        # ICI approval line
        ax.axvline(x=2016.5, color=colors["ici"], linestyle="--",
                   linewidth=1.5, alpha=0.7)
        ax.text(2016.7, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 5,
                "ICI\napproval", fontsize=8, color=colors["ici"],
                va="top", ha="left")

        # Trend lines
        for d_grp, color in [(d_pre, colors["pre"]), (d_post, colors["post"])]:
            if len(d_grp) >= 3:
                slope, intercept, r, p, se = stats.linregress(d_grp["year"], d_grp[col])
                x_fit = np.array([d_grp["year"].min(), d_grp["year"].max()])
                ax.plot(x_fit, intercept + slope * x_fit, "--",
                        color=color, alpha=0.5, linewidth=1)

        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Year of Diagnosis", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2003, 2022)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/figures/figure1_trends.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nFigure 1 saved to: outputs/figures/figure1_trends.png")

# ─── FIGURE 2: ITS PLOTS ─────────────────────────────────────────────────────

if plot_outcomes:
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(
        "Interrupted Time Series Analysis: Effect of ICI Availability\n"
        "on Bladder Cancer Treatment Utilization and Stage at Diagnosis",
        fontsize=13, fontweight="bold", y=1.01
    )
    axes2 = axes2.flatten()

    for idx, (col, label) in enumerate(plot_outcomes):
        ax  = axes2[idx]
        model, d = its_model(annual_df, col)
        if model is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.scatter(d[d["ici_era"]==0]["year"], d[d["ici_era"]==0][col],
                   color="#2E75B6", zorder=5, s=40, label="Observed (pre)")
        ax.scatter(d[d["ici_era"]==1]["year"], d[d["ici_era"]==1][col],
                   color="#C00000", zorder=5, s=40, marker="s", label="Observed (post)")
        ax.plot(d["year"], model.fittedvalues, "k-", linewidth=2, label="ITS fitted")
        ax.axvline(x=2016.5, color="#888888", linestyle="--", linewidth=1.5, alpha=0.7)

        # Annotate level change
        beta2 = model.params.get("ici_era", np.nan)
        p2    = model.pvalues.get("ici_era", np.nan)
        if not np.isnan(beta2):
            sign = "+" if beta2 >= 0 else ""
            ax.text(0.05, 0.05,
                    f"Level change: {sign}{beta2:.1f}%\np = {p2:.4f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.8))

        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Year of Diagnosis", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/figures/figure2_its.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved to: outputs/figures/figure2_its.png")

print("\nNote: For formal joinpoint regression with APC/AAPC statistics,")
print("import outputs/annual_trends.csv into the NCI Joinpoint Program.")
print("Download free from: surveillance.cancer.gov/joinpoint")
print("\nNext step: run 04_survival_analysis.py")
