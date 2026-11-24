# Part 2 Report: What We Did and Why

**Project:** Evidence-Aware Classification of AI-Linked Workforce Events Using Machine Learning
**Notebook:** `Part2_Modelling_and_Evaluation.ipynb` · **Demo:** `app.py`

This document walks through Part 2 step by step. For every step it says **what** was done, **why** it was done that way, and **what came out of it**. It follows Phases 3–6 of our submitted proposal. Section numbers in brackets refer to the notebook.

---

## 0. Where Part 1 left us

Part 1 produced three things Part 2 depends on:

| Part 1 output | Why it matters for Part 2 |
|---|---|
| `ai_layoffs_clean.csv` (320 events, cleaned + engineered features) | The modelling data. |
| A **company-grouped split** (`sgkf_fold`): fold 0 = test (75 events), folds 1–4 = train (245) | Repeat companies (a third of the data) can never be in both train and test, so the model cannot score well by memorising employers. |
| `feature_spec.json`: feature lists for the 3 experiments, leakage exclusions, TF-IDF settings | Part 2 rebuilds exactly the same features, without leaking label-derived columns. |

---

## 1. Reusable feature module [Section 3.1]

**What.** We copied Part 1's cleaning and feature-engineering code (company-name masking, text cleaning, sector/country mapping, % of workforce parsing, job-family flags, engineered text counts) into `src/attribution_features.py`. The notebook runs this module on the **raw** CSV and asserts that all 38 model columns match the Part 1 output exactly.

**Why.** The demo must process a *new* event exactly as the training data was processed. If the code were duplicated by hand, a small difference (a regex, a mapping) would silently change predictions — called *training/serving skew*. The assertion proves there is none (0 mismatches).

---

## 2. Data splitting: train / validation / test [Section 4]

**What.**

| Set | Size | Use |
|---|---|---|
| Training (folds 1–4) | 245 | Fitting models |
| Validation | 4-fold grouped cross-validation inside the 245 | Tuning, comparing and selecting models |
| Test (fold 0) | 75 | Used **once**, after the final model was chosen |
| Chronological hold-out | 64 most recent events | Robustness check only |

We turned the Part 1 fold column into a scikit-learn splitter with `PredefinedSplit`, and asserted that no company crosses a fold boundary.

**Why.**
- A single validation set of ~50 events would contain only ~6 *explicit* events; one prediction would move the score by several points. Cross-validation uses every training event for validation once and averages out this noise.
- Reusing the Part 1 folds keeps the **group-aware** property (proposal: "use group-aware splitting where related records could otherwise appear in both training and testing").
- Keeping the test set untouched until the end gives an honest estimate of performance on unseen companies.

---

## 3. Preprocessing pipeline [Section 5]

**What.** A function `make_preprocessor()` rebuilds the Part 1 `ColumnTransformer`:
- **Text:** TF-IDF (1–2-grams, English stop words, `min_df=2`, sublinear TF) → optional `SelectKBest(chi2, k)`.
- **Text statistics** (AI terms, business reasons, third-party words, quotes, …) and **context numbers** (log headcount, % of workforce, year, month sin/cos, …): median imputation + scaling.
- **Context categories** (sector group, region, country, kind, execution): one-hot encoding (rare values grouped).
- **Binary flags:** passed through.

The whole preprocessor is the first step of each model `Pipeline`.

**Why.**
- Because it sits *inside* the pipeline, every TF-IDF vocabulary, scaler, encoder and χ² selection is re-learned on each CV training part only → **no leakage** from validation/test data (proposal Phase 2: "fit preprocessing transformations using training data only").
- Naive Bayes needs non-negative inputs, so it uses `MinMaxScaler` instead of `StandardScaler`; gradient boosting needs a dense matrix. These are options of the same function, so all models share identical features otherwise.
- The text configs produce ~2,100 features for 245 events ("p ≫ n"), which is why regularised models and feature selection were important.

---

## 4. Model selection [Section 6]

**What.** Six models, exactly as proposed plus one extra ensemble:

| Group | Model | Why chosen |
|---|---|---|
| Reference | Majority class | The floor any model must beat (macro-F1 = 0.25). |
| Traditional | Multinomial Naive Bayes | Classic text classifier; very few parameters → good on small data. |
| Traditional | Logistic Regression | Regularised and interpretable (per-class coefficients for RQ1). |
| Traditional | Linear SVC | Often the strongest linear model for sparse TF-IDF text. |
| Ensemble (bagging) | Random Forest | Mixes text and tabular features, robust to noisy features. *Added so RQ3 compares two ensemble families.* |
| Ensemble (boosting) | HistGradientBoosting | scikit-learn's gradient boosting ("gradient boosting where feasible" in the proposal; XGBoost was optional and not needed). |

**Metric:** **macro-F1** (primary, as in the proposal), plus accuracy, balanced accuracy, macro precision/recall, per-class F1 and confusion matrix.

**Why macro-F1?** Classes are imbalanced 199 : 82 : 39. Accuracy would reward always predicting *mixed* (60 % accuracy!). Macro-F1 gives each class equal weight, so the model must also get *explicit* and *blamed* right.

---

## 5. Training and hyperparameter tuning [Section 7]

**What.** `GridSearchCV` for each of 6 models × 3 feature configurations (18 searches), scoring macro-F1 on the grouped 4-fold CV. Tuned settings:

| Model | Hyperparameters tuned |
|---|---|
| Naive Bayes | `alpha`, `fit_prior`, χ² `k` |
| Logistic Regression | `C`, χ² `k` |
| Linear SVC | `C`, χ² `k` |
| Random Forest | `max_depth`, `min_samples_leaf`, `max_features`, χ² `k` (500 trees) |
| Gradient Boosting | `learning_rate`, `max_iter`, χ² `k` (`max_depth=3`, `min_samples_leaf=10`) |

**Result (mean CV macro-F1):**

| Model | Text only | Context only | Combined |
|---|---|---|---|
| Majority | 0.256 | 0.256 | 0.256 |
| Naive Bayes | 0.553 | 0.468 | 0.592 |
| Logistic Regression | 0.536 | 0.494 | 0.554 |
| Linear SVC | 0.514 | 0.497 | 0.561 |
| **Random Forest** | 0.618 | 0.510 | **0.665** |
| Gradient Boosting | 0.497 | 0.454 | 0.528 |

**Why grid search with CV?** It is the systematic way to choose hyperparameters without touching the test set (proposal: "tune using training data only"). Grids were kept small because with 245 events large searches mainly find settings that are lucky on the CV folds.

**Practical note:** gradient boosting is itself multi-threaded, so its search runs with `n_jobs=1`; running both in parallel oversubscribed the CPU and took ~10× longer.

---

## 6. Optimisation and performance improvement [Section 8]

| Experiment | What we did | Result | Decision |
|---|---|---|---|
| **Class weighting** | Re-ran tuned models with and without `class_weight="balanced"` | Helps all models; Random Forest **collapses** without it (macro-F1 0.31, explicit recall 0) | Keep weighting everywhere |
| **χ² feature selection** | CV score for k = 50 … all n-grams | k = 50–100 hurts all; RF and NB peak at k ≈ 1,000 (RF drops to 0.605 with all n-grams); LR likes all | Tune `k` per model (done in the grid) |
| **Soft-voting ensemble** | Averaged probabilities of RF + NB (+ LR) | 0.59–0.60, worse than the single RF (0.665) | Not used |
| **Seed stability** | RF with 5 random seeds | 0.635–0.665 | Result is stable; ±0.02 differences are noise |
| **Learning curve** | CV score vs. training size | RF still rising (0.45 → 0.63), LR flat; training ≈ 1.0 | Models are data-limited: more labelled events would help most |

**Why these?** They are the levers the proposal listed under "Potential challenges": class imbalance → class weighting; small data / overfitting → feature selection and complexity control; ensemble methods → voting; and checking that results are stable rather than lucky.

---

## 7. Final model selection [Section 9]

**Rule (fixed before looking at the test set):** highest mean grouped-CV macro-F1.

**Selected:** **Random Forest on combined features** — 500 trees, `max_depth=10`, `min_samples_leaf=3`, `max_features=0.05`, χ² top-1000 n-grams, balanced class weights. CV macro-F1 = **0.665 ± 0.039**.

**Why it makes sense:** best CV score, uses both text and context (RQ2), stable across seeds, complexity is limited by depth/leaf size/feature sub-sampling, and it outputs class probabilities for the demo.

---

## 8. Test-set evaluation [Section 10]

| Metric (75 unseen-company events) | Final model | Majority reference |
|---|---|---|
| Macro-F1 | **0.570** (95 % bootstrap CI 0.44–0.70) | 0.250 |
| Balanced accuracy | 0.597 | 0.333 |
| Accuracy | 0.653 | 0.600 |
| F1 explicit / mixed / blamed | 0.33 / 0.72 / 0.65 | 0 / 0.75 / 0 |

**Why a bootstrap confidence interval?** With 75 events (only 7 explicit), one prediction can change the score a lot. The CI shows the real uncertainty: the model is clearly above the baseline, but its exact score is only known to about ±0.13.

**Why is the test score lower than CV, and why not pick the model that won on test?** The text-only Linear SVC happened to score 0.615 on this test set but only 0.514 in CV. All test scores fall inside the final model's CI, so the test set cannot reliably rank models. Choosing the model by its test score would be "selecting on the test set" and would overstate real performance. CV (245 events) is the more reliable basis for selection; the test set is for an honest final estimate.

---

## 9. Comparisons: answering RQ2 and RQ3 [Section 11]

**RQ2 — text vs. context vs. combined.**
- Text ≫ context (best CV 0.618 vs 0.510).
- Combined > text only in CV for **all five** models (+0.02 to +0.05).
- On the test set the gain is inconsistent (−0.06 to +0.05) and every paired bootstrap CI includes 0.
- **Answer:** adding metadata gives a modest, consistent CV improvement, but the evidence is not strong enough to call it a reliable gain on new companies. The claim wording does most of the work.

**RQ3 — traditional vs. ensemble.**
- **Random Forest (bagging) is best** in every configuration in CV.
- **Gradient boosting is not better** than the traditional models — with ~180 events per CV training part, boosting overfits.
- **Naive Bayes** is a remarkably strong simple baseline.
- **Answer:** "ensemble" is not automatically better; averaging (bagging) helps on this small, noisy dataset, sequential boosting does not.

---

## 10. Robustness checks [Section 12]

| Check | Why | Result |
|---|---|---|
| **Remove register "verdict" phrases** (`text_clean_noverdict`) | Part 1 found 13 claims with the register's own verdict wording ("does not mention AI"), mostly *blamed*. If the model relied on them, it would be reading the label. | CV 0.665 → 0.654, test 0.570 → 0.567: **no dependence** on these phrases. |
| **Chronological hold-out** (train on older 256, test on newest 64) | Proposal: "consider a chronological holdout to examine performance on more recent events". | Macro-F1 0.669 (0.679 on the 53 events from new companies), majority 0.256. **No sign of decay**, but only 3 explicit events in this period. |
| **Probability calibration** (sigmoid) | Proposal: "show model confidence where supported and appropriately calibrated". | Calibration improved log-loss (0.79 → 0.70) but dropped macro-F1 to 0.505 → **not used**; the demo shows raw probabilities as *relative* confidence and says so. |

---

## 11. Interpretation: answering RQ1 [Section 13]

Three views were used because each has a weakness alone:
1. **Permutation importance** (model-agnostic, per original input column): shuffling the text costs ≈ 0.10 macro-F1, ~3× any other column.
2. **Random Forest importance:** TF-IDF n-grams hold ≈ 61 % of total importance; context blocks ≈ 27 %; engineered text stats the rest.
3. **Logistic Regression coefficients** (per-class, interpretable): *explicit* ← "automated", "agents", "shift ai", "founder"; *blamed* ← "press", "named", "framed", "chatgpt"; *mixed* ← "growth", "platform", "built". Strongest context signal: event kind = *workforce statement* → explicit.

**Answer to RQ1:** the wording of the claim (who speaks, and whether AI is the stated reason or a reported link) is the most informative evidence. Context features are weak individually but useful as a group (sector, event kind, whether size figures were reported).

---

## 12. Error analysis [Section 14]

- 26 of 75 test events are misclassified; **22 of the 26 involve the *mixed* class** (mixed → explicit 8; mixed ↔ blamed 14). These are the boundaries the register itself has to judge.
- Typical case: IBM's CEO saying HR workers were replaced with AI agents reads *explicit*, but the register labels it *mixed* because other reasons were also given elsewhere. A model that sees only the summary text cannot know that.
- *explicit → blamed* (3 of 7 explicit events): short claims where the AI link is told in reported speech.
- Confidence is meaningful: 0.56 on correct vs 0.47 on wrong predictions; the most confident quarter is 89 % accurate. → The demo flags low-margin predictions for manual review.

**Why do error analysis?** The proposal (Phase 5) asks to "investigate confusion between closely related attribution categories" and "discuss limitations of automated classification when evidence is incomplete or ambiguous". It also tells us *what* to improve: more explicit examples.

---

## 13. Final ML pipeline and demo [Section 15, `app.py`]

```
raw event fields → attribution_features.engineer_features()  (masking, cleaning, feature engineering)
               → ColumnTransformer (TF-IDF + χ² | scaled numbers | one-hot context | flags)
               → RandomForestClassifier (tuned, class-weighted)
               → predicted category + confidence per class
```

- Saved as one object: `models/final_pipeline.joblib` (+ `model_metadata.json` with hyperparameters, metrics, versions and the disclaimer).
- **Re-fitted on all 320 events** after evaluation. Why: once selection and evaluation are finished, using all data gives a better model (the learning curve is still rising). The reported performance is still the honest test estimate of the train-only model.
- An end-to-end check reloads the file from disk and classifies three invented events: CEO blog → explicit, bank results with cost + AI → mixed, union/press report → blamed.
- **Streamlit demo** (`streamlit run app.py`): user enters claim, source, metadata → app shows predicted category, confidence bars, a low-margin warning, the features the model saw, and the disclaimer that predictions are **not proof of AI-caused layoffs** (proposal Phase 6).

---

## 14. How the proposal was covered

| Proposal item | Where |
|---|---|
| Baselines: majority, Multinomial NB, Logistic Regression | Sections 6–7 |
| Comparative: Linear SVC, gradient boosting | Sections 6–7 (+ Random Forest) |
| Experiments 1–3 (text / context / combined) | Sections 7, 11 |
| Stratified, group-aware split; CV on training data; chronological hold-out | Sections 4, 12 |
| Macro-F1, precision, recall, per-class F1, balanced accuracy, confusion matrix | Section 10 |
| Class weighting, macro-F1, per-class evaluation for imbalance | Sections 8.1, 10 |
| Leakage control, training-only preprocessing | Sections 3.1, 5 |
| Interpretation, error analysis, minority-class effect | Sections 13, 14 |
| Streamlit demo with confidence and disclaimer | Section 15, `app.py` |
| Git / GitHub | README §8 |

## 15. Limitations (be ready to discuss these)

- **Small and imbalanced:** 320 events, 39 explicit → wide confidence intervals; small differences between models are not meaningful.
- **Labels are one organisation's judgement** of what was said, not the true cause of the layoffs.
- **The text is the register's own summary**, so the model partly learns its writing style and sources.
- **Probabilities are not calibrated** (calibration hurt classification).
- **Time:** no decay seen yet, but only 3 recent explicit events; the model should be re-trained as the register grows.

---

## Viva summary (Roman Urdu)

- **Kya banaya:** ek classifier jo kisi layoff event ke text aur context se predict karta hai kay AI link *explicit* (company ne khud AI ko wajah kaha), *mixed* (AI kai wajohat mein se ek) ya *blamed* (company ne AI nahi kaha, press/union ne kaha) hai.
- **Split:** Part 1 ka company-grouped split use kiya — test (75) mein koi aisi company nahi jo training mein ho. Validation ke liye 4-fold grouped cross-validation, kyunki 245 events mein alag validation set bohat chhota hota.
- **Models:** Majority (baseline), Naive Bayes, Logistic Regression, Linear SVC (traditional), Random Forest aur Gradient Boosting (ensemble) — teen feature sets (text, context, combined) par, GridSearchCV se tune kiye.
- **Metric:** macro-F1, kyunki classes imbalanced hain (199/82/39); accuracy sirf "mixed" bol kar bhi 60% aa jati.
- **Optimisation:** class weighting (Random Forest is ke baghair fail ho jata hai), χ² se 1000 best n-grams, voting ensemble try kiya (faida nahi hua), seeds aur learning curve check kiye.
- **Final model:** Random Forest + combined features, CV macro-F1 0.665; test par 0.570 (CI 0.44–0.70), baseline 0.25 se double se zyada.
- **RQ1:** claim ka text sab se important hai (kaun bol raha hai, AI wajah hai ya reported link). **RQ2:** text + context CV mein har model ke liye behtar, lekin test par significant nahi. **RQ3:** Random Forest (bagging) best, boosting chhote data par overfit karta hai.
- **Robustness:** verdict phrases hata kar bhi same result (leakage nahi), naye events par bhi performance theek; calibration ne classification kharab ki is liye use nahi ki.
- **Errors:** zyada tar galtiyan *mixed* ke saath (mixed↔explicit, mixed↔blamed) — yeh wohi cases hain jo insaan ke liye bhi mushkil hain.
- **Demo:** `streamlit run app.py` — naya event daalo, category + confidence milti hai, aur disclaimer kay yeh AI ki wajah se layoff ka saboot nahi.
