# Telco Customer Churn Dataset - Data Dictionary

## Dataset Overview
- **Source:** IBM Telco Customer Churn dataset (original source: Kaggle / IBM)
- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records:** 7,043 customers
- **Columns:** 21 (1 identifier + 20 features)
- **Target Variable:** `Churn` (whether customer left within last month)

---

## Column Details

| Column | Type | Description | Missing Values |
|--------|------|-------------|----------------|
| `customerID` | String | Unique customer identifier | 0 |
| `gender` | Categorical | Male / Female | 0 |
| `SeniorCitizen` | Integer (0/1) | Whether customer is a senior citizen (1 = Yes, 0 = No) | 0 |
| `Partner` | Categorical | Whether customer has a partner (Yes / No) | 0 |
| `Dependents` | Categorical | Whether customer has dependents (Yes / No) | 0 |
| `tenure` | Integer | Number of months the customer has stayed with the company | 0 |
| `PhoneService` | Categorical | Whether customer has phone service (Yes / No) | 0 |
| `MultipleLines` | Categorical | Whether customer has multiple lines (Yes / No / No phone service) | 0 |
| `InternetService` | Categorical | Customer's internet service (DSL / Fiber optic / No) | 0 |
| `OnlineSecurity` | Categorical | Whether customer has online security (Yes / No / No internet service) | 0 |
| `OnlineBackup` | Categorical | Whether customer has online backup (Yes / No / No internet service) | 0 |
| `DeviceProtection` | Categorical | Whether customer has device protection (Yes / No / No internet service) | 0 |
| `TechSupport` | Categorical | Whether customer has tech support (Yes / No / No internet service) | 0 |
| `StreamingTV` | Categorical | Whether customer streams TV (Yes / No / No internet service) | 0 |
| `StreamingMovies` | Categorical | Whether customer streams movies (Yes / No / No internet service) | 0 |
| `Contract` | Categorical | Contract term (Month-to-month / One year / Two year) | 0 |
| `PaperlessBilling` | Categorical | Whether customer has paperless billing (Yes / No) | 0 |
| `PaymentMethod` | Categorical | Customer's payment method (Electronic check / Mailed check / Bank transfer / Credit card) | 0 |
| `MonthlyCharges` | Float | Amount charged to customer monthly | 0 |
| `TotalCharges` | Float (stored as String) | Total amount charged to customer over entire tenure | 11 whitespace-only values |
| `Churn` | Categorical (Target) | Whether customer churned (Yes / No) | 0 |

---

## Target Variable Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| No (No Churn) | 5,174 | 73.5% |
| Yes (Churn) | 1,869 | 26.5% |

**Note:** The dataset is moderately imbalanced with ~3:1 ratio of non-churn to churn customers.

---

## Known Data Quality Issues
1. **TotalCharges:** 11 records contain whitespace instead of numeric values (these correspond to customers with 0 tenure)
2. **SeniorCitizen:** Stored as integer (0/1) but semantically represents a boolean flag
3. **Contract:** Has ordinal nature (Month-to-month < One year < Two year)
4. **MultipleLines, OnlineSecurity, etc.:** Include "No internet service" as a category for customers without internet

---

## Source Attribution
Original dataset from IBM Watson Studio, accessible via Kaggle.
Dataset URL: https://www.kaggle.com/datasets/blastchar/telco-customer-churn