# Credit-Card-Fraud-Detection-Project
This project implements a machine learning pipeline to detect fraudulent credit card transactions using **XGBoost**. It includes hyperparameter optimization with **Optuna**, model evaluation, class balancing, and explainability with **SHAP**.

> **Dataset:** This project uses the **Credit Card Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📌 Project Overview

The goal of this project is to identify fraudulent credit card transactions from a dataset stored in a MySQL database. The workflow includes:

1. **Data Loading** – Fetching transaction data directly from a MySQL database.
2. **Exploratory Data Analysis (EDA)** – Inspecting missing values, feature distributions, correlation analysis, and fraud rates by numeric and categorical features.
3. **Feature Importance Analysis** – Using Random Forest with nested cross-validation and Optuna-based hyperparameter tuning to identify the most predictive features.
4. **Model Training & Optimization** – Training XGBoost models with Optuna hyperparameter tuning on both original and balanced training sets.
5. **Balanced Training** – Handling imbalanced datasets by upsampling minority classes to ensure fair learning.
6. **Model Explainability** – Generating SHAP summary plots to interpret feature contributions to model predictions.
7. **Evaluation** – Assessing models using metrics such as F1-score, ROC-AUC, precision, recall, and fraud rate consistency.

---

## 🛠️ Technologies & Libraries

- **Python 3.9+**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **SQLAlchemy** – Database connection
- **scikit-learn** – Machine learning models, preprocessing, and metrics
- **XGBoost** – Gradient boosting model for classification
- **Optuna** – Hyperparameter optimization framework
- **SHAP** – Model explainability and feature contribution
- **Matplotlib / Seaborn** – Data visualization

---
