# Diabetes Risk Prediction using Machine Learning

##📌 Project Overview##
This project builds a machine learning model to predict whether a patient is likely to have diabetes based on health indicators such as age, BMI, blood glucose level, HbA1c level, smoking history, and more. The goal is to demonstrate an end-to-end ML workflow — from data cleaning to model evaluation and interpretation.

## 📊 Dataset
- **Source:** Diabetes Prediction Dataset (Kaggle)
- **Size:** 100,000 patient records
- **Features:** gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
- **Target:** `diabetes` (0 = No, 1 = Yes)
- **Class distribution:** ~91.5% No Diabetes, ~8.5% Diabetes (imbalanced dataset)

## 🛠️ Tech Stack
- Python
- Pandas, NumPy — data handling
- Matplotlib, Seaborn — visualization
- Scikit-learn — preprocessing, modeling, evaluation
- Google Colab — development environment

## 🔍 Steps Followed

### 1. Data Cleaning
- Removed duplicate rows
- Removed the rare `Other` gender category (only 18 rows)

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution to identify imbalance
- Compared age, BMI, HbA1c, and blood glucose distributions across diabetic vs non-diabetic patients
- Explored diabetes rates by gender and smoking history
- Generated a correlation heatmap of numeric features

### 3. Train/Test Split
- Split data 80/20 using stratified sampling to preserve class balance in both sets

### 4. Preprocessing Pipeline
- Standardized numeric features (age, bmi, HbA1c_level, blood_glucose_level)
- One-hot encoded categorical features (gender, smoking_history)
- Built using `ColumnTransformer` + `Pipeline` to prevent data leakage

### 5. Model Training
- Trained a **Random Forest Classifier** with `class_weight='balanced'` to handle class imbalance

### 6. Model Evaluation
- Evaluated using precision, recall, F1-score, and ROC-AUC (not just accuracy, due to class imbalance)
- Visualized results with a confusion matrix and ROC curve

### 7. Feature Importance
- Identified the most influential features in predicting diabetes
- Top predictors: HbA1c level and blood glucose level — consistent with real-world clinical knowledge

## 📈 Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.958 |
| Precision (class 1) | 0.93 |
| Recall (class 1) | 0.69 |
| F1-score (class 1) | 0.80 |
| Accuracy | 0.97 |
## 🖼️ Sample Visualizations
- Class distribution
- Feature distributions by outcome
- Correlation heatmap
- Confusion matrix
- ROC curve
- Feature importance chart

*(Add screenshots of your plots here once uploaded)*

## ⚠️ Limitations
- Dataset is synthetic/Kaggle-sourced, not real clinical data — results wouldn't generalize directly to real-world deployment without further validation
- No hyperparameter tuning was performed in this basic version (potential next step)

## 🚀 Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Try additional models (XGBoost, Logistic Regression comparison)
- SHAP-based model explainability
- Deploy as a simple web app (Streamlit)

## 📂 How to Run
1. Clone this repository
2. Open `diabetes_prediction.ipynb` in Google Colab or Jupyter Notebook
3. Upload `diabetes_prediction_dataset.csv` to the same environment
4. Run all cells in order

## 👤 Author
*(NIRMALA)*

ML model to predict diabetes risk using patient health data
