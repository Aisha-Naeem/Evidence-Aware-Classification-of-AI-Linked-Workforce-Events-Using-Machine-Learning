# Evidence-Aware Classification of AI-Linked Workforce Events Using Machine Learning

**DSTT Term Project: Part 1 (Data Preparation & EDA) and Part 2 (Modelling, Evaluation & ML Pipeline)**
Data Science Tools and Techniques (DSTT), MS Data Science
Group: **Ayesha Naeem (26L-8007)** · **Muhammad Mazhar (26L-8027)**

---

## 1. Project overview

Employers increasingly mention artificial intelligence when they announce layoffs, but how strongly they tie a cut to AI varies a lot. This project builds a **multi-class classifier** that predicts how the AI link of a workforce event was established, using the event's text and context.

| Label (`Attribution`) | Meaning | Events |
|---|---|---|
| **explicit** | The employer itself declared the cut AI-related. | 39 |
| **mixed** | The employer cited AI *among other reasons*. | 199 |
| **blamed** | The employer named no AI reason; press, unions or analysts linked it to AI. | 82 |

**Research questions**
- **RQ1:** Which textual and contextual features are most informative for separating the three categories?
- **RQ2:** Does combining text with contextual metadata (sector, country, event kind, …) beat text alone?
- **RQ3:** How do traditional classifiers compare with ensemble methods?

| Notebook | What it covers |
|---|---|
| `Part1_Data_Preparation_and_EDA.ipynb` | Loading and verifying the dataset snapshot, cleaning (types, missing values, duplicates, category harmonisation, outliers), text preprocessing with company-name masking, feature engineering, a **leakage audit**, a **company-grouped train/test split**, a train-only preprocessing pipeline for three feature configurations, feature relevance and 18 EDA figures. No models are trained. |
| `Part2_Modelling_and_Evaluation.ipynb` | Train/validation/test design (grouped CV), 6 models × 3 feature configurations with hyperparameter tuning, optimisation (class weighting, χ² feature selection, voting ensemble, seed stability, learning curve), final model selection on CV, one-time test evaluation with bootstrap CIs, model comparison (RQ2, RQ3), robustness checks (chronological hold-out, verdict-phrase removal, calibration), interpretation (RQ1), error analysis, and the saved end-to-end pipeline. 13 more figures (fig19–fig31). |
| `app.py` | Streamlit demo that classifies a new, user-entered event with the saved pipeline. |

> The labels record the register's classification of *what was said* about each event. They are **not** proof that AI caused any job losses, and neither are this project's predictions.

### Headline results (Part 2)

| | Value |
|---|---|
| Final model | **Random Forest** on combined text + context features (χ² top-1000 n-grams, `max_depth=10`, `min_samples_leaf=3`, `max_features=0.05`, balanced class weights) |
| Grouped 4-fold CV macro-F1 (245 training events) | **0.665** (± 0.039 across folds) |
| Held-out test macro-F1 (75 events, unseen companies) | **0.570**, 95 % bootstrap CI [0.44, 0.70]; balanced accuracy 0.597; accuracy 0.653 |
| Majority-class reference (test macro-F1) | 0.250 |
| Per-class test F1 (explicit / mixed / blamed) | 0.33 / 0.72 / 0.65 |
| RQ1 | Claim wording is the most informative input; context helps as a group. |
| RQ2 | Text ≫ context; combining them raises CV macro-F1 for all five models (+0.02–0.05), but the gain is not significant on the small test set. |
| RQ3 | Bagging (Random Forest) is best; boosting does not beat the traditional models on 245 events; Naive Bayes is a strong simple baseline. |

The full explanation of every step, and why it was done, is in **[`PROJECT_REPORT_Part2.md`](PROJECT_REPORT_Part2.md)**.

## 2. Dataset

| | |
|---|---|
| Name | **AI Layoffs Register** |
| Publisher | AI Layoffs Index, <https://ailayoffs.org> (methodology: <https://ailayoffs.org/methodology>) |
| Kaggle | <https://www.kaggle.com/datasets/ailayoffs/ai-layoffs-register-2023-2026-every-row-sourced> |
| Version used | Methodology **v1.36**, snapshot **2026-10-23** |
| Download mirror (exact snapshot) | <https://raw.githubusercontent.com/adrianobuilder/demissoes_ia/main/raw/ai-layoffs-register.csv> |
| Size | **320 events × 20 columns**, dates 2023-02-22 to 2026-10-22 |
| Companies | **257** distinct companies (256 after merging *Telstra* / *Telstra International*); 228 distinct `Sector` values; 50 `Country` values |
| Classes | mixed 199 · blamed 82 · explicit 39 |

We deliberately do **not** use the live endpoint (`/api/register.csv`). It keeps growing, which would make the results irreproducible. A copy of the snapshot is in `data/raw/`. Part 1 downloads it from the mirror only if the file is missing.

**Licence note.** The register is published by AI Layoffs Index and redistributed on Kaggle. Check the licence shown on the Kaggle dataset page before redistributing the data, and cite AI Layoffs Index as the source. The copy here is included for academic reproducibility only.

## 3. Repository structure

```
DSTT_Term_Project/
├── Part1_Data_Preparation_and_EDA.ipynb   # Part 1 notebook (saved with outputs)
├── Part2_Modelling_and_Evaluation.ipynb   # Part 2 notebook (saved with outputs)
├── app.py                                 # Streamlit demo
├── PROJECT_REPORT_Part2.md                # what was done in Part 2 and why
├── README.md
├── requirements.txt
├── src/
│   └── attribution_features.py            # Part 1 feature engineering as a reusable module (used by Part 2 and the app)
├── data/
│   ├── raw/ai-layoffs-register.csv        # fixed 320-row snapshot (unchanged)
│   └── processed/
│       ├── ai_layoffs_clean.csv           # cleaned + engineered columns, split columns
│       ├── feature_spec.json              # feature lists per experiment, exclusions, split settings
│       └── preprocessor_combined.joblib   # combined ColumnTransformer fitted on the training split (Part 1)
├── models/
│   ├── final_pipeline.joblib              # final end-to-end scikit-learn pipeline (Part 2)
│   └── model_metadata.json                # hyperparameters, metrics, versions, disclaimer
├── results/                               # CSV tables written by Part 2 (CV results, test results, robustness, importances, summary)
└── figures/
    ├── fig01_ … fig18_*.png               # Part 1 figures
    └── fig19_ … fig31_*.png               # Part 2 figures
```

## 4. Required libraries

| Library | Used for |
|---|---|
| pandas, numpy | data handling and numerical operations |
| matplotlib, seaborn | plotting |
| scikit-learn | TF-IDF, encoders, `ColumnTransformer`, `Pipeline`, `GridSearchCV`, `PredefinedSplit`, Naive Bayes, Logistic Regression, Linear SVC, Random Forest, HistGradientBoosting, calibration, metrics, permutation importance |
| scipy | Cramér's V and Kruskal–Wallis tests (Part 1) |
| joblib | saving the fitted preprocessor and the final pipeline |
| jupyter, nbconvert | running the notebooks |
| streamlit | the demo app |

Tested with Python 3.10.11, pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10, seaborn 0.13.2. The saved `models/final_pipeline.joblib` should be loaded with the same scikit-learn version it was trained with (see `models/model_metadata.json`); if you use another version, re-run Part 2 to re-create it.

## 5. Setup

**Option A: venv**
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**Option B: conda**
```bash
conda create -n dstt python=3.10 -y
conda activate dstt
pip install -r requirements.txt
```

## 6. How to run

All paths are relative, so **run everything from the project root** (the folder that contains this README).

1. **Part 1:** open `Part1_Data_Preparation_and_EDA.ipynb` → *Kernel → Restart & Run All* (under a minute). It writes `data/processed/*` and `figures/fig01–fig18`.
2. **Part 2:** open `Part2_Modelling_and_Evaluation.ipynb` → *Kernel → Restart & Run All* (about 7–10 minutes on a laptop; the grid searches dominate). It writes `models/*`, `results/*` and `figures/fig19–fig31`.
3. **Demo:** `streamlit run app.py`, then open the URL it prints (usually <http://localhost:8501>).

Headless alternative:
```bash
jupyter nbconvert --to notebook --execute --inplace Part1_Data_Preparation_and_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Part2_Modelling_and_Evaluation.ipynb
```

Both notebooks contain assertions that stop them if the data does not match the expected snapshot, if train and test share a company, or (Part 2) if the reusable feature module stops reproducing the Part 1 features exactly. All random seeds are fixed (`random_state=42`), so re-running gives the same numbers. On Windows, joblib may print harmless `resource_tracker ... KeyError` messages in the terminal during headless execution; they do not affect the results.

## 7. Key design decisions

**Part 1**
- **No rows dropped.** With 320 events and only 39 explicit ones, problems are fixed by recoding, flagging or (train-only) imputation.
- **Leakage audit.** Excluded from all features: `What the employer said` (Cramér's V = 0.95, restates the label), `Why not counted`, `Evidence tier`, `Counts in US index`, and the counted / not-counted headcount split. The two headcount columns are merged into one `headcount`.
- **Group-aware split, made before any fitting.** `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by normalised company: fold 0 = test (75), folds 1–4 = train (245). A `chrono_split` column marks the most recent ~20 % of events.
- **Company-name masking** before vectorising, so models learn attribution language rather than company names.
- **Three feature configurations:** Experiment 1 = text (TF-IDF 1–2-grams + engineered text statistics); Experiment 2 = context only; Experiment 3 = both.

**Part 2**
- **Validation = grouped 4-fold CV** over the Part 1 training folds (`PredefinedSplit` on `sgkf_fold`), because a single validation set from 245 events would contain only ~6 explicit events.
- **Everything is fitted inside the CV pipeline** (TF-IDF, χ² selection, encoders, imputers, scalers), so no validation information leaks into training.
- **Primary metric: macro-F1**, with class weighting for all models that support it (without it, Random Forest collapses to predicting *mixed*).
- **Final model chosen on CV only;** the test fold is used once, and all models' test scores are reported with a bootstrap CI for transparency.
- **Same features in training and serving:** `src/attribution_features.py` is asserted to reproduce the Part 1 features exactly from the raw CSV, and the demo uses that module.
- **Deployed model** = the selected pipeline re-fitted on all 320 events after evaluation; the reported performance is the test estimate of the train-only model.

## 8. Uploading to GitHub

```bash
git init
git add .
git commit -m "DSTT term project: Part 1 and Part 2"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
All files are small (largest is the 2.6 MB model), so no Git LFS is needed. Submit the repository link together with the Part 2 notebook.
