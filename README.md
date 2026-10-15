# Bank Marketing Classification Pipeline 🏦📊

An end-to-end machine learning project designed to predict whether a bank customer will subscribe to a term deposit (`y`), using advanced ensemble classifiers and hyperparameter optimization.

---

## 🚀 Project Overview
This repository features a comparative machine learning study using the classic **Bank Marketing** dataset. The dataset exhibits severe class imbalance (~89% negative class vs. ~11% positive class), making it an ideal candidate for evaluating various classification strategies. 

The pipeline explores and optimizes three major tree-based models:
1. **Random Forest Classifier** (`RandomizedSearchCV`)
2. **Gradient Boosting Classifier** (`RandomizedSearchCV`)
3. **XGBoost Classifier** (Optimized via both `RandomizedSearchCV` and **Optuna**)

---

## 🛠️ Data Preprocessing & Pipeline
Data transformation is handled uniformly using Scikit-Learn's `ColumnTransformer` and embedded into a final scikit-learn `Pipeline`:
* **Ordinal Encoding:** Applied to the `education` column using a logical custom hierarchy (`illiterate` $\rightarrow$ `university.degree`).
* **One-Hot Encoding:** Applied to categorical features (`job`, `marital`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) with unknown categories ignored.
* **Standard Scaling:** Applied to numerical/continuous variables (`age`, `duration`, `campaign`, `pdays`, `previous`, economic indicators, etc.).
* **Feature Engineering:** Added a binary `previously_contacted` feature derived from the `pdays` column.

---

## 📈 Model Performance Summary

| Model | Hyperparameter Search Method | Test ROC-AUC | Class 1 (Yes) Recall | Class 1 (Yes) F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | RandomizedSearchCV | 0.9467 | 0.36 | 0.48 |
| **Gradient Boosting** | RandomizedSearchCV | 0.9471 | 0.51 | 0.57 |
| **XGBoost** | RandomizedSearchCV | 0.9475 | 0.71 | 0.64 |
| **XGBoost (Optuna)** | Bayesian Optimization (100 Trials) | **0.9482** | **0.79** | **0.64** |

---

## 🔍 Key Features of the Final Optuna-Tuned XGBoost Model
The final model integrates an optimized set of hyperparameters found via **Optuna** (`TPESampler`) maximizing the cross-validation `f1` score, alongside `scale_pos_weight` to address class imbalance effectively:

```python
best_opt_model = XGBClassifier(
    n_estimators=642,
    max_depth=5,
    learning_rate=0.025,
    subsample=0.722,
    colsample_bytree=0.953,
    min_child_weight=0.754,
    reg_alpha=0.397,
    reg_lambda=0.960,
    scale_pos_weight=3,
    random_state=42,
    eval_metric="logloss",
)
