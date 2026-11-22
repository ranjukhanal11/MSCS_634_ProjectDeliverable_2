# MSCS_634_Project_Deliverable_2

**Project:** Advanced Data Mining for Data-Driven Insights and Predictive Modeling  
**Deliverable:** 2 — Regression Modeling and Performance Evaluation  
**Author:** Ranju Khanal, Anusha Sharma
**Institution:** University of the Cumberlands — MSCS-634-M50

---

## Dataset and Source

- **Name:** Breast Cancer Wisconsin (Diagnostic) (WDBC), UCI Machine Learning Repository.
- **UCI original:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Reproducible access:** `sklearn.datasets.load_breast_cancer(as_frame=True)` — runs without any external download.
- **Original dataset:** 569 observations, 30 numerical measurements, one binary diagnosis label.
- **Selected subset (from Deliverable 1):** 10 mean nuclear measurements + `diagnosis`.
- **Regression target:** `mean area` (explicitly excluded from the input features).
- **Stratified split:** 80% train (455 records) / 20% test (114 records), `random_state = 42`.

---

## Objective

Build and compare at least two regression models to predict `mean area` from the
remaining nine mean measurements, using feature engineering, five-fold
cross-validation, and multiple evaluation metrics (R², MSE, RMSE, MAE). Investigate
multicollinearity, interpret the near-perfect R² via a geometric-identity explanation,
and validate the modeling framework on a genuinely harder target
(`mean fractal dimension`).

---

## Repository Layout

```text
MSCS_634_Project_Deliverable_2/
├── README.md
├── requirements.txt
├── Deliverable_2_Regression_Modeling.ipynb
├── data/
│   └── wdbc_cleaned.csv
└── figures/
    ├── 01_target_distribution.png
    ├── 02_predicted_vs_actual.png
    ├── 03_cv_r2_boxplot.png
    ├── 04_coefficients_top10.png
    └── 05_learning_curve.png
```

---

## How to Run

1. Install Python 3.10+ and Jupyter Notebook / JupyterLab.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `Deliverable_2_Regression_Modeling.ipynb` and select **Run All**.
4. The notebook loads the dataset through scikit-learn, builds and evaluates the
   three regression models, and saves all figures to `figures/`.
5. No external dataset download is required.

---

## Models Implemented

| Model | Regularization | Purpose |
|---|---|---|
| Linear Regression | None | Baseline |
| Ridge (α = 1.0) | L2 | Variance control under multicollinearity |
| Lasso (α = 0.01) | L1 | Automatic feature selection |

All models are wrapped in `Pipeline(StandardScaler → model)` so preprocessing is
fit only on training folds. Evaluation uses 5-fold cross-validation on the training
set; the held-out test set is evaluated once at the end.

---

## Key Results

### Primary regression — target: `mean area`

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Linear Regression | **0.9993** | 8.54 | 5.84 |
| Lasso (α = 0.01) | **0.9993** | 8.55 | 5.75 |
| Ridge (α = 1.0) | 0.9987 | 11.54 | 8.12 |

Cross-validated R² (5-fold, training set): all three models in the range **0.9976–0.9983**
with fold-to-fold standard deviation under 0.0011.

### Ablation — removing `radius_squared` and `perimeter_to_radius`

| Model | Full R² | Reduced R² | Full RMSE | Reduced RMSE |
|---|---|---|---|---|
| Linear Regression | 0.9993 | 0.9891 | 8.54 | 32.94 |
| Lasso (α = 0.01) | 0.9993 | 0.9891 | 8.55 | 32.90 |
| Ridge (α = 1.0) | 0.9987 | 0.9888 | 11.54 | 33.39 |

R² remains ≈ 0.989 even without the geometric shortcut, driven by the strong
multicollinearity between `mean radius`, `mean perimeter`, and `mean area`
(r ≈ 0.987 – 0.998).

### Secondary regression — target: `mean fractal dimension`

| Model | CV R² | Test R² | Test RMSE |
|---|---|---|---|
| Linear Regression | 0.8455 | 0.8325 | 0.0030 |
| Ridge (α = 1.0) | 0.8438 | 0.8328 | 0.0030 |
| Lasso (α = 0.005) | **−0.0024** | −0.0031 | 0.0074 |

Lasso with an over-aggressive penalty collapsed to the mean predictor — an
instructive negative result demonstrating that regularization is not always helpful.

---

## Key Insights

1. **R² ≈ 1.0 is a warning, not a success.** `mean area ≈ π · radius²` for
   near-circular masses, so the model is re-deriving a geometric identity rather
   than discovering new biological signal.
2. **Multicollinearity is structural.** Radius, perimeter, and area are nearly
   linear combinations of one another; raw coefficients in the OLS model are not
   individually interpretable.
3. **Regularization is conditional.** Ridge and Lasso offered no benefit on the
   geometric shortcut task, and an over-aggressive Lasso collapsed entirely on
   the harder secondary target.
4. **Leakage-safe pipelines work.** Cross-validation and held-out test scores
   agree closely across all models, confirming no preprocessing leakage.

---

## Figures

| Figure | Description |
|---|---|
| `figures/01_target_distribution.png` | Histogram + box plot of `mean area` |
| `figures/02_predicted_vs_actual.png` | Predicted vs actual for all three models |
| `figures/03_cv_r2_boxplot.png` | 5-fold CV R² distribution per model |
| `figures/04_coefficients_top10.png` | Top-10 standardized coefficients (OLS) |
| `figures/05_learning_curve.png` | Learning curve for Ridge regression |

---

## Challenges and Resolutions

| Challenge | Resolution |
|---|---|
| Risk of target leakage | Explicit exclusion of `mean area` from features via list comprehension |
| Multicollinearity among size features | Systematic ablation study rather than arbitrary deletion |
| Right-skewed target | Documented; log-target suggested as future work |
| Over-aggressive Lasso on secondary target | Reported as instructive negative result; alpha tuning deferred |
| Shared mutable state across notebook cells | `sklearn.base.clone()` used in ablation loop |

---



