# bladder-cancer-rwe

**Stage at Diagnosis, Surgical Treatment Utilization, and Survival Outcomes in Muscle-Invasive and Metastatic Bladder Cancer Before and After Immune Checkpoint Inhibitor Availability**

*A SEER Population-Based Analysis of Racial, Socioeconomic, and Geographic Disparities, 2004–2021*

---

[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)](.)
[![Data](https://img.shields.io/badge/Data-SEER%20Public-blue)](https://seer.cancer.gov)
[![Language](https://img.shields.io/badge/Language-Python%203.10+-green)](.)
[![IRB](https://img.shields.io/badge/IRB-Exempt-lightgrey)](.)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This repository contains the complete, reproducible analysis pipeline for **Paper 1** of a three-paper real-world evidence (RWE) portfolio examining treatment patterns and outcomes in bladder cancer.

The study uses **publicly available SEER data** (free registration, no institutional affiliation required) to examine how immune checkpoint inhibitor (ICI) approval in 2016–2017 changed the treatment landscape and survival outcomes in muscle-invasive and metastatic urothelial carcinoma — and which patient populations were left behind.

### Why this study matters to pharma and payers

> The primary outcome — **proportion of patients receiving no cancer-directed treatment** — is the foundational market access evidence gap every bladder cancer drug manufacturer needs quantified. Identifying which patient characteristics predict no-treatment receipt across racial, socioeconomic, and geographic subgroups directly informs coverage expansion arguments, patient assistance program design, and formulary positioning strategies.

---

## Research Questions

1. What proportion of patients with MIBC/metastatic bladder cancer received **no cancer-directed treatment**, and how did this vary by race, rurality, income, and age before vs. after ICI approval (2016–2017)?
2. How did first-line surgical and radiation treatment receipt shift following ICI availability, and which patient subgroups were **systematically excluded** from the treatment era benefit?
3. What is the median time from diagnosis to first treatment initiation by stage, race, and county-level rurality — and did this change in the ICI era?
4. What are 1-, 3-, and 5-year **overall survival (OS) and cancer-specific survival (CSS)** trends by treatment received, stage, and demographic subgroup before vs. after ICI availability?

---

## Repository Structure

```
bladder-cancer-rwe/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── protocol/
│   └── Paper1_Study_Protocol.docx    ← Full formal study protocol (12 sections)
│
├── data/
│   └── seer_raw.csv                  ← Place SEER*Stat export here (not committed)
│
├── src/
│   ├── 01_cohort_identification.py   ← Cohort build, variable engineering, audit log
│   ├── 02_descriptive_table1.py      ← Table 1 with SMD and chi-square tests
│   ├── 03_trend_analysis.py          ← ITS models + NCI Joinpoint data export
│   ├── 04_survival_analysis.py       ← KM, Fine-Gray competing risk, Cox (3 models)
│   ├── 05_logistic_no_treatment.py   ← Primary market access outcome analysis
│   └── 06_sensitivity_analyses.py    ← 5 pre-specified sensitivity analyses
│
└── outputs/
    ├── figures/                       ← Publication-ready figures (300 DPI PNG)
    ├── table1.csv
    ├── cox_results.csv
    ├── table3_no_treatment_logistic.csv
    ├── annual_trends.csv              ← Import into NCI Joinpoint Program
    ├── its_results.csv
    ├── km_results.csv
    └── sensitivity_results.csv
```

---

## Data Source

**SEER Research Data, 17 Registries, Nov 2023 Submission (2000–2021)**

- Free registration: [seer.cancer.gov/data](https://seer.cancer.gov/data)
- No institutional affiliation required
- Download SEER*Stat software (free): [seer.cancer.gov/seerstat](https://seer.cancer.gov/seerstat)
- IRB: Exempt under 45 CFR 46.102(l) — publicly de-identified data

### How to get the data (15 minutes)

1. Go to [seer.cancer.gov/data](https://seer.cancer.gov/data) → "Request SEER Data"
2. Complete registration (email + intended use description)
3. Receive credentials within 24–48 hours
4. Open SEER*Stat → connect to SEER Research server
5. Case Listing Session → select SEER 17 Registries 2000–2021
6. Filter: Primary Site = C670–C679 (bladder)
7. Add all variables listed in `src/01_cohort_identification.py` COLUMN_MAP
8. Export as CSV → save to `data/seer_raw.csv`

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/brad-ekwu-ochigbo/bladder-cancer-rwe.git
cd bladder-cancer-rwe

# Install dependencies
pip install -r requirements.txt

# Place your SEER export at data/seer_raw.csv, then run in order:
python src/01_cohort_identification.py
python src/02_descriptive_table1.py
python src/03_trend_analysis.py
python src/04_survival_analysis.py
python src/05_logistic_no_treatment.py
python src/06_sensitivity_analyses.py
```

Each script prints a progress log and tells you what to run next.

---

## Statistical Methods

| Method | Purpose | Script |
|---|---|---|
| Descriptive statistics + SMD | Table 1 by ICI era | `02` |
| Joinpoint regression | Annual trend APC/AAPC | `03` + NCI software |
| Interrupted time series (Newey-West SE) | Pre/post ICI level and slope change | `03` |
| Kaplan-Meier + log-rank | OS/CSS by subgroup | `04` |
| Fine-Gray competing risk | Cancer vs. other-cause death | `04` |
| Multivariable Cox (3 models) | Adjusted HR for OS | `04` |
| Multivariable logistic regression | Predictors of no treatment | `05` |
| Sensitivity analyses (5 pre-specified) | Robustness assessment | `06` |

---

## Expected Outputs

| Output | Description |
|---|---|
| Table 1 | Patient characteristics by ICI era (pre vs. post) |
| Table 2 | Treatment receipt by era, race, rurality, income |
| Table 3 | Multivariable logistic regression — no treatment receipt |
| Table 4 | OS and CSS at 1, 3, 5 years by subgroup |
| Table 5 | Cox regression results (3 models) |
| Figure 1 | Annual trend plots with joinpoint annotations |
| Figure 2 | Interrupted time series — treatment utilization |
| Figure 3 | Kaplan-Meier curves by race/ethnicity and ICI era |
| Figure 4 | Cumulative incidence functions (competing risk) |
| Figure 5 | Forest plot — Cox subgroup analysis |
| Figure 6 | No-treatment rate heatmap by race × rurality × era |

---

## Study Design

| Element | Specification |
|---|---|
| Design | Retrospective population-based cohort study |
| Guidelines | STROBE + RECORD statements |
| Population | Adults with MIBC/metastatic urothelial carcinoma |
| Index date | Date of incident bladder cancer diagnosis |
| Pre-ICI period | January 2004 – December 2015 |
| Transition year | 2016 (excluded — FDA approval transition year) |
| Post-ICI period | January 2017 – December 2021 |
| Primary outcomes | Overall survival, cancer-specific survival |
| Key secondary outcome | No cancer-directed treatment receipt (market access outcome) |

---

## Target Journals

1. **Value in Health** (primary) — ISPOR's journal; read by pharma HEOR teams
2. **Cancer Medicine** — open access, fast review, PubMed indexed
3. **Urologic Oncology** — specialty audience, bladder cancer clinicians and industry
4. **JNCI Cancer Spectrum** — NCI-affiliated, open access, disparity focus

---

## Paper Portfolio Context

This is Paper 1 of a three-paper bladder cancer RWE portfolio:

| Paper | Study | Data | Status |
|---|---|---|---|
| **P1 (this repo)** | Natural history, treatment patterns, survival | SEER public | 🟡 In progress |
| P2 | HCRU and economic burden by treatment line | Claims | 🔵 Planned |
| P3 | Comparative effectiveness EV+pembro vs. chemo — target trial emulation | Claims | 🔵 Planned |

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
lifelines>=0.27.0
statsmodels>=0.13.0
openpyxl>=3.0.0
```

---

## Citation

*[To be added upon publication]*

---

## License

Code: MIT License
Data: Subject to [SEER Data Use Agreement](https://seer.cancer.gov/data/terms.html)

---

## Author

**Brad Ekwu-Ochigbo, RPh, PhD**
RWE Analyst | Oncology HEOR | Pharmacoeconomics
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/ekwu-b-ochigbo-rph-phd-43ab5264)
