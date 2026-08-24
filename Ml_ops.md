# ML_OPS.md — Data & Model Architecture Specification
## BomaMicro Finance — Credit Risk & Default Prediction Engine

**Author:** BATHROMEO PROTAS SYOKO

---

## 1. Class Imbalance & Data Quality

Baseline profiling of `boma_loans_raw.csv` (n=10,000) confirms an 18.0% default rate against 82.0% performing loans, consistent with the client-reported portfolio deterioration. Data quality issues identified: 494 nulls in `monthly_income_tzs`, 300 nulls in `days_late_avg`, 251 records with negative income (physically invalid), and a cluster of `age = 150` values indicative of a sentinel/placeholder artifact rather than true outliers.

**Accuracy is explicitly rejected as the primary metric.** At an 82/18 base rate, a degenerate classifier that predicts "no default" universally achieves 82% accuracy with zero recall on the minority class — the exact failure mode this engagement exists to prevent, given the asymmetric cost structure (a missed defaulter is materially more expensive than a false-positive rejection). **ROC-AUC and Recall are the governing metrics**, with Precision-Recall AUC tracked as a secondary check since it's more informative than ROC-AUC under this degree of imbalance.

**Imbalance mitigation, applied in escalating order:**

1. **Class-weight adjustment** (`class_weight='balanced'` / `scale_pos_weight`) as the default intervention — no synthetic data, no distributional distortion, minimal complexity cost.
2. **SMOTE**, conditional on class-weighting alone being insufficient at the recall threshold — applied *exclusively within the training fold, post-split*, to prevent synthetic minority samples from being derived using information that leaks from the validation/test partitions.
3. **Focal loss** reserved as a fallback if the architecture shifts to a neural network; for tree ensembles, class-weighting achieves the equivalent effect with less implementation risk.

**Data quality remediation:** regional-median imputation for missing income and delinquency fields (income distributions plausibly vary by region — Dar es Salaam, Mwanza, Arusha, Mbeya, Dodoma — so a global mean would be a biased estimator here); negative-income and `age=150` records are flagged and corrected rather than dropped, to avoid introducing selection bias if the corruption isn't missing-at-random. All cleaning decisions are validated against the raw distribution to ensure they aren't inflating downstream performance metrics artificially.

---

## 2. Model Transparency & Fairness

Regulatory constraints preclude a pure black-box deployment; the model must produce a decision *and* a defensible rationale per applicant. **SHAP (SHapley Additive exPlanations)** is the selected explainability framework, for three reasons: (1) native compatibility with tree ensembles (XGBoost/Random Forest, the anticipated production candidates), (2) instance-level attribution — required to generate the "Top 3 risk factors" per applicant, not merely a global feature-importance ranking, and (3) consistency guarantees rooted in cooperative game theory, meaning identical inputs yield identical explanations — a property LIME's local surrogate approximations don't reliably provide, and one that matters when explanations may face regulatory scrutiny.

The Logistic Regression baseline is interpretable by construction (coefficient inspection), which also serves as a sanity check against the SHAP output on the more complex model — material divergence between the two would itself be a signal worth investigating before deployment.

Implementation: SHAP values are computed per prediction, ranked by absolute contribution, and mapped to human-readable statements (e.g., "elevated prior default count," "income below regional median") rather than exposed as raw feature names or numeric attributions — the API consumer is a loan officer or customer, not a data scientist.

---

## 3. Data Leakage & Train/Test Splitting

Leakage risk is assessed at three levels:

- **Target leakage**: confirming `days_late_avg` and `previous_defaults` are computed strictly from information available at decision time, with no contamination from outcome data associated with the loan being scored.
- **Preprocessing leakage**: all imputers, scalers, and encoders are fit exclusively on the training partition and applied (never refit) to the test partition — fitting on the full dataset prior to splitting is a common and consequential error that silently inflates validation metrics.
- **Resampling leakage**: SMOTE, if used, is applied strictly post-split on the training fold; pre-split oversampling risks synthetic points derived from test-set minority instances propagating information into training.
- **Temporal leakage**: this snapshot lacks transaction-level timestamps, but if raw time-series mobile-money data is incorporated later, a **chronological split** (train on historical loans, validate on subsequent loans) supersedes random splitting — random partitioning of time-adjacent financial data allows the model to implicitly learn macro-level effects (e.g., a regional economic shock) that wouldn't generalize to genuinely future applicants.

**Splitting protocol**: stratified train/test split preserving the 82/18 class ratio in both partitions, with stratified k-fold cross-validation during model selection and hyperparameter tuning — a single split is an unreliable performance estimate at this minority-class prevalence. The test partition remains untouched until final evaluation.

---

## 4. Inference Latency & Deployment

Sub-1-second inference is not the binding constraint for tree-based models; deployment architecture is.

1. **Decouple production logic from exploratory code**: cleaning, feature engineering, inference, and explanation logic are extracted into versioned modules (`src/preprocessing.py`, `src/features.py`, `src/model.py`). The notebook remains an EDA/experimentation artifact and is never imported by the serving layer.
2. **Artifact serialization** via `joblib` (preferred over `pickle` for numpy-backed scikit-learn/XGBoost objects) — the model and its full preprocessing pipeline are persisted as a single deployable artifact, eliminating any re-fitting at request time.
3. **Service layer: FastAPI**, chosen over Flask for native Pydantic request validation (non-negotiable for a financial-decision endpoint — malformed input is rejected before reaching the model) and async request handling under concurrent load.
4. **API contract**: a single `POST /predict` endpoint accepting applicant JSON, executing the persisted preprocessing/feature pipeline, and returning `{risk_score, decision, top_3_risk_factors}`.
5. **Warm-start architecture**: model and SHAP explainer are loaded once at process startup, not per-request — disk I/O on the hot path, not model inference itself, is the dominant latency risk.
6. **Stretch scope**: containerization via Docker for deployment portability, and structured prediction logging (input, output, timestamp) to establish the audit trail regulatory compliance will require.