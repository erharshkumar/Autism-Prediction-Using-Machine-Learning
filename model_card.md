
# Model Card: Autism Screening Classifier

**Version:** Tier 4 (Final)  |  **Maintainer:** Harsh Kumar (harshkumar.in)  |  **Date:** 2025

---

## 1. Model Description

| Field | Value |
|---|---|
| Task | Binary classification — ASD screening |
| Architecture | Stacking ensemble (DT + RF + XGBoost → Logistic Regression meta-learner) |
| Input features | 15 (AQ-10 items A1–A10 + age, gender, ethnicity, jaundice, family autism history, country, prior app use, relation) |
| Output | P(ASD) ∈ [0, 1] + binary screen label |
| Framework | scikit-learn + imblearn + XGBoost + Optuna |
| Serialised artifact | `autism_pipeline_tier4.pkl` + `autism_threshold_tier4.json` |

---

## 2. Intended Use

### ✅ Primary intended use
- Preliminary ASD screening to help clinicians **prioritise** patients for formal diagnostic evaluation
- Research into ML-based neurodevelopmental screening tools
- Educational demonstration of an end-to-end ML pipeline

### ❌ Out-of-scope uses
- **NOT** a diagnostic tool — does not replace formal clinical assessment by a licensed practitioner
- **NOT** validated for children under 4 (AQ-10 is designed for adults)
- **NOT** suitable as the sole basis for any educational, legal, or medical decision
- **NOT** approved for clinical deployment without external validation and regulatory clearance

---

## 3. Training Data

| Property | Value |
|---|---|
| Source | AQ-10 Autism Screening Dataset (Kaggle) |
| Size | 800 rows, 15 features (after feature selection) |
| Features removed | `ID` (identifier), `age_desc` (constant), `result` (derived from AQ-10 items — circular) |
| Target | `Class/ASD` — 0 = no autism, 1 = autism |
| Class balance | Class 0: 639 (79.9%) | Class 1: 161 (20.1%) |
| Train / test split | 80 / 20 stratified |
| Class imbalance handling | SMOTE (Synthetic Minority Oversampling) on training fold only |
| Countries covered | 54 |
| Ethnicities covered | 10 categories |
| Age range | 2–89 years |

---

## 4. Evaluation Results

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | 0.9158 | Ranking quality across all thresholds |
| PR-AUC | 0.6829 | Precision-Recall area; more informative on imbalanced data |
| Sensitivity | 0.9375 | % of ASD cases correctly identified (recall for positive class) |
| Specificity | 0.7891 | % of non-ASD correctly cleared |
| Precision | 0.5263 | Of those flagged, % truly have ASD |
| F1 (ASD class) | 0.6742 | Harmonic mean of precision and recall |
| Classification threshold | 0.1426 | Optimised for sensitivity ≥ 0.90 via PR curve |
| Missed ASD cases (FN) | 2 / 32 | At optimal threshold on test set |

---

## 5. Hyperparameter Optimisation

| Aspect | Detail |
|---|---|
| Method | Optuna (Tree-structured Parzen Estimator) |
| Trials | 50 per base model (150 total) |
| CV folds | 5-fold stratified |
| Primary metric | ROC-AUC |
| Search space | 6–7 hyperparameters per model |

---

## 6. Ethical Considerations

### 6.1 Diagnostic disclaimer
This model produces a **screening result**, not a diagnosis. A positive screen indicates
the person should be referred for comprehensive evaluation by a qualified clinician.
The model's output should never be communicated to a patient as a diagnosis.

### 6.2 Training data biases
- The dataset was collected via a self-report screening app; respondents self-selected,
  which may not represent the general population
- ASD is historically **underdiagnosed in females** and some minority ethnic groups —
  training labels may encode this historical bias
- The dataset has limited representation of some ethnic groups (< 10 test samples for 5 of 10 groups)

### 6.3 Privacy
- The model processes inputs in memory and produces predictions — no patient data should be persisted
- The deployment environment must comply with applicable health data regulations (HIPAA, GDPR, DISHA)

---

## 7. Limitations

1. **Small dataset** (800 samples) — all metrics carry wide confidence intervals
2. **Single dataset** — model has not been externally validated on a different population
3. **Self-report bias** — AQ-10 responses may not accurately reflect autism traits
4. **Country feature** — country of residence is used as a proxy for unobserved confounders; this may introduce spurious correlations
5. **No temporal validation** — model trained on data from a single time period; autism diagnostic criteria evolve
6. **SMOTE limitation** — synthetic oversampling can create unrealistic minority-class samples; real data collection is preferable

---

## 8. Fairness Summary

*Full analysis in Tier 3 notebook.*

| Group | n (test) | Sensitivity | Note |
|---|---|---|---|
| Male | ~32 | — | See Tier 3 fairness analysis |
| Female | ~14 | — | Small positive-class sample; estimate unreliable |
| White-European | ~51 | — | Largest group; most reliable estimate |
| Others | ~47 | — | Mixed group; interpret with caution |
| Remaining ethnicities | < 20 each | — | Estimates unreliable at this sample size |

**Recommendation:** Before any deployment, conduct fairness evaluation on a dataset with at least
50 true-positive cases per demographic subgroup.

---

## 9. Citation

If using this model for research:

```
Harsh Kumar (2025). Autism Prediction ML Pipeline (Tier 4).
github.com/harshkumar/autism-prediction
harshkumar.in
```
