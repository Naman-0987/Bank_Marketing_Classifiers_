# Bank Direct Marketing Campaign – Machine Learning Pipeline

This notebook builds an end-to-end Machine Learning pipeline to predict whether a client will subscribe to a term deposit (`y` = yes/no) based on marketing campaign data. 

To address heavy class imbalance (~11% positive cases), we benchmark individual classifiers and combine them into a tuned **Soft Voting Ensemble**.

## 🛠️ Pipeline Architecture
1. **Data Preprocessing & Feature Engineering**:
   - `OrdinalEncoder`: Encodes ordered categorical features (`education`).
   - `OneHotEncoder`: Encodes nominal categorical features (`job`, `marital`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`).
   - `StandardScaler`: Scales continuous numerical features (`age`, `duration`, `campaign`, `pdays`, `euribor3m`, etc.).
   - Feature Creation: Transformed `pdays` into `previously_contacted`.
2. **Model Training & Hyperparameter Tuning**:
   - **Decision Tree**: Cost-sensitive learning (`class_weight='balanced'`) tuned via `RandomizedSearchCV`.
   - **Logistic Regression**: Regularized classifier (`liblinear`) tuned via `RandomizedSearchCV`.
   - **Support Vector Machine (SVC)**: Non-linear kernel (`rbf`) with calibrated output probabilities.
3. **Ensemble Modeling**:
   - **Soft Voting Classifier**: Combines predictions from all three base models to maximize **ROC-AUC (0.9423)**.

---
