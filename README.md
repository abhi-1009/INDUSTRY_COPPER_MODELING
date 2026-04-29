## Industrial Copper Modeling
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.x-FF6600?style=flat&logo=xgboost)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Pickle](https://img.shields.io/badge/Model%20Serialization-Pickle-yellow?style=flat)

An end-to-end **Machine Learning application** for the industrial copper market that predicts:
1. **Selling Price** — Regression model predicting the exact selling price
2. **Transaction Status** — Classification model predicting Won / Lost outcome
The app uses advanced preprocessing (log transformation, IQR outlier clipping, SMOTETomek oversampling), trains multiple ML models, serialises them with Pickle, and serves predictions through an interactive two-tab Streamlit dashboard.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Model Accuracy Results](#model-accuracy-results)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [References](#references)

## Project Overview
The industrial copper market suffers from data skewness, outliers, and class imbalance — all of which degrade raw model performance. This project addresses all three challenges through a rigorous preprocessing pipeline before training dual ML models:
- **Regression** — predicts `selling_price` (log-transformed target, inverse-transformed for output)
- **Classification** — predicts `status` (Won=1 / Lost=0) with SMOTETomek oversampling to fix class imbalance
Both trained models are saved as `.pkl` files and loaded in the Streamlit app for real-time user predictions.

## Dataset
| Property | Detail |
| :--- | :--- |
| **File** | `Copper_Set.xlsx` |
| **Domain** | Industrial copper transactions |
| **Target (Regression)** | `selling_price` (continuous) |
| **Target (Classification)** | `status` → Won (1) / Lost (0) |
### Key Features Used for Prediction
| Feature | Type | Notes |
| :--- | :---: | :--- |
| `customer` | Numeric | Customer ID |
| `country` | Numeric | Encoded country code |
| `status` | Categorical | Mapped to 0–8 |
| `item type` | Categorical | Ordinal encoded (W, WI, S, PL, IPL, SLAWR, Others) |
| `application` | Numeric | Application code |
| `width` | Numeric | IQR outlier clipped |
| `product_ref` | Numeric | Product reference code |
| `quantity tons` | Numeric | Log-transformed |
| `thickness` | Numeric | Log-transformed |
| `item_date` | Date | Split into day/month/year |
| `delivery date` | Date | Split into day/month/year; wrong dates corrected |
### Columns Dropped
- `id` — all unique, no predictive value
- `material_ref` — >55% null values; values starting with `00000` treated as null

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, cleaning, type conversion |
| **NumPy** | latest | Log transformation, array operations |
| **Scikit-Learn** | 1.x | Preprocessing, ML models, metrics, GridSearchCV |
| **XGBoost** | 1.x | XGBRegressor and XGBClassifier |
| **imbalanced-learn** | latest | SMOTETomek — oversample minority class |
| **Matplotlib** | latest | EDA visualisations |
| **Seaborn** | latest | Distribution and correlation plots |
| **Pickle** | built-in | Save and load trained models |
| **Streamlit** | 1.x | Two-tab interactive prediction dashboard |
| **datetime** | built-in | Date parsing, delivery date correction |

### Python Libraries (from source code)
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import warnings
```

## Data Preprocessing Pipeline
```
Copper_Set.xlsx
      │
      ▼
1. Type Conversion
   ├── quantity tons → numeric (coerce errors)
   ├── item_date / delivery date → datetime → date
   └── material_ref: values starting with '00000' → NaN

      │
      ▼
2. Drop Columns
   └── Drop: id, material_ref (>55% null / all unique)

      │
      ▼
3. Handle Invalid Values
   ├── quantity tons ≤ 0 → NaN
   └── selling_price ≤ 0 → NaN

      │
      ▼
4. Fill Null Values
   ├── Object columns (item_date, status, delivery date) → mode
   └── Numeric columns (quantity, customer, country, application,
       thickness, selling_price) → median

      │
      ▼
5. Encode Categorical Variables
   ├── status → map (Lost:0, Won:1, Draft:2 ... Offerable:8)
   └── item type → OrdinalEncoder

      │
      ▼
6. Log Transformation (handle skewness)
   ├── quantity tons → quantity tons_log
   ├── thickness → thickness_log
   └── selling_price → selling_price_log

      │
      ▼
7. Outlier Handling — IQR clip() method
   ├── quantity tons_log
   ├── thickness_log
   ├── selling_price_log
   └── width
   (Values clipped to [Q1 - 1.5*IQR, Q3 + 1.5*IQR])

      │
      ▼
8. Delivery Date Correction
   ├── Calculate Date_difference = delivery_date - item_date
   ├── Negative differences → recalculate delivery date
   └── Split dates into day / month / year columns

      │
      ▼
9. Classification: SMOTETomek Oversampling
   └── Balance Won (1) vs Lost (0) class imbalance

      │
      ▼
10. Train Models → Save as .pkl
    ├── regression_model.pkl  (predict selling_price)
    └── classification_model.pkl (predict Won/Lost)
```

## Machine Learning Models
### Regression Models Evaluated (predict `selling_price`)
| Model | Purpose |
| :--- | :--- |
| Decision Tree Regressor | Baseline tree model |
| Extra Trees Regressor | Ensemble of extremely randomised trees |
| **Random Forest Regressor** | Selected for deployment |
| AdaBoost Regressor | Boosting on weak learners |
| Gradient Boosting Regressor | Sequential boosting |
| XGBoost Regressor | Extreme gradient boosting |
### Classification Models Evaluated (predict `status`: Won/Lost)
| Model | Purpose |
| :--- | :--- |
| Decision Tree Classifier | Baseline tree model |
| Extra Trees Classifier | Ensemble of extremely randomised trees |
| **Random Forest Classifier** | Selected for deployment |
| AdaBoost Classifier | Boosting on weak learners |
| Gradient Boosting Classifier | Sequential boosting |
| XGBoost Classifier | Extreme gradient boosting |
### Final Deployed Model Parameters (from source code)
```python
# Classification — Random Forest
RandomForestClassifier(
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2
)
```
## Model Accuracy Results
### Regression — Selling Price Prediction
| Metric | Value |
| :--- | :---: |
| **Model** | Random Forest Regressor |
| **R² Score** | ~0.95+ |
| **Evaluation** | Mean Absolute Error (MAE), MSE, R² |
| **Target** | `selling_price_log` (inverse exp() transform on output) |
| **Train/Test Split** | 80% / 20%, `random_state=42` |
> Output is inverse log-transformed: `selling_price = np.exp(y_pred[0])`, rounded to 2 decimal places.

### Classification — Won / Lost Status Prediction
| Metric | Value |
| :--- | :---: |
| **Model** | Random Forest Classifier |
| **Accuracy** | ~96%+ |
| **Oversampling** | SMOTETomek (fixes Won vs Lost class imbalance) |
| **Evaluation** | Accuracy, Confusion Matrix, Classification Report, ROC-AUC |
| **Train/Test Split** | 80% / 20%, `random_state=42` |
### Status Encoding Reference
| Status | Encoded Value |
| :---: | :---: |
| Lost | 0 |
| Won | 1 |
| Draft | 2 |
| To be approved | 3 |
| Not lost for AM | 4 |
| Wonderful | 5 |
| Revised | 6 |
| Offered | 7 |
| Offerable | 8 |
### Item Type Encoding Reference
| Item Type | Encoded Value |
| :---: | :---: |
| IPL | 0.0 |
| Others | 1.0 |
| PL | 2.0 |
| S | 3.0 |
| SLAWR | 4.0 |
| W | 5.0 |
| WI | 6.0 |

## Streamlit Dashboard
The app has **two tabs** — one for each prediction task:
### Tab 1 — Predict Selling Price (Regression)
| Input Field | Type | Range |
| :--- | :---: | :--- |
| Item Date | Date picker | 2020-07-01 to 2021-05-31 |
| Delivery Date | Date picker | 2020-08-01 to 2022-02-28 |
| Quantity Tons | Text input | Min: 0.00001, Max: 1,000,000,000 |
| Customer ID | Text input | Min: 12,458,000, Max: 2,147,484,000 |
| Country | Selectbox | 17 country codes |
| Status | Selectbox | 9 status options |
| Item Type | Selectbox | 7 types |
| Application | Selectbox | 30 application codes |
| Thickness | Number input | 0.1 to 2,500,000 |
| Width | Number input | 1.0 to 2,990,000 |
| Product Ref | Selectbox | 33 product references |
**Output:** `Predicted Selling Price = ₹XXXX.XX` (green, centered) + 🎈 balloons

### Tab 2 — Predict Status (Classification)
Same inputs as Tab 1, with `Selling Price` replacing `Status`.
**Output:**
- Won → `Predicted Status = Won` (green) + 🎈 balloons
- Lost → `Predicted Status = Lost` (green) + ❄️ snow

## Installation and Setup
### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/Industrial-Copper-Modeling.git
cd Industrial-Copper-Modeling
```
### Step 2 — Install Required Libraries
```bash
pip install streamlit pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn openpyxl
```
### Step 3 — Add Dataset
Place `Copper_Set.xlsx` in the project folder and update the path:
```python
ICM = 'Copper_Set.xlsx'
```
### Step 4 — Train Models and Generate Pickle Files
```bash
python copper_modeling.py
```
This generates two files in your project folder:
- `regression_model.pkl`
- `classification_model.pkl`

### Step 5 — Launch the Streamlit App
```bash
streamlit run copper_app.py
```
## Usage
**Tab 1 — Predict Selling Price:**
1. Fill in all fields (Item Date, Delivery Date, Quantity Tons, Customer ID, Country, Status, Item Type, Application, Thickness, Width, Product Ref)
2. Click **SUBMIT**
3. Predicted selling price appears below in green
**Tab 2 — Predict Status:**
1. Fill in all fields (same as Tab 1, with Selling Price replacing Status)
2. Click **SUBMIT**
3. Result shows **Won** or **Lost** with animation

> If Quantity Tons or Customer ID is left empty, a warning message is displayed instead of crashing.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [imbalanced-learn SMOTETomek](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
