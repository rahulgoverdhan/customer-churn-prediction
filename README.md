# Customer Churn Prediction & Business Analytics

## Overview
End-to-end data science project predicting customer churn with interpretable ML models and an interactive Streamlit dashboard.

## Problem Statement
Identify customers likely to churn within 30-60 days and provide business insights for retention strategies.

## Dataset
Kaggle Telco Customer Churn dataset (7,043 records, 21 features)

## Project Structure
```
customer-churn-predict/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned/engineered features
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_ingest.py    # Load and clean data
│   ├── features.py       # Feature engineering
│   ├── model.py          # Model training and evaluation
│   ├── interpret.py      # SHAP/interpretability
│   ├── app.py            # Streamlit web app
│   └── utils.py          # Helper functions
├── reports/              # Generated reports and images
├── models/               # Saved trained models
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Setup Instructions

### 1. Clone and install dependencies
```bash
git clone <repo-url>
cd customer-churn-predict
pip install -r requirements.txt
```

### 2. Run EDA
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 3. Train models
```bash
python src/model.py
```

### 4. Launch dashboard
```bash
streamlit run src/app.py
```

## Key Findings (to be documented)
- Month-to-month contracts have highest churn rate
- Fiber optic internet customers churn more frequently
- Electronic check payment method correlates with churn
- Low tenure = high churn risk

## Model Performance
- ROC AUC: >0.80 (target)
- Recall for churn class: priority metric

## License
MIT