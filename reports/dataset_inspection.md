# Dataset Inspection Results - Telco Customer Churn

## Basic Properties
- **Rows:** 7,043
- **Columns:** 21
- **File Size:** 970,457 bytes (~970 KB)

## Missing Values
- **Total Missing Values:** 0
- **TotalCharges:** 11 whitespace-only values (stored as empty strings/spaces)
- All other columns: 0 missing values

## Duplicate Rows
- **Duplicate Rows Found:** 0

## Target Variable Distribution (Churn)
```
No:  5,174 (73.5%)
Yes: 1,869 (26.5%)
```

## Data Types
- **Numeric:** tenure (int64), SeniorCitizen (int64), MonthlyCharges (float64)
- **String/Object:** All other columns (categorical features)
- **Note:** TotalCharges is stored as object (string) due to whitespace values

## Feature Categories
1. **Demographics:** gender, SeniorCitizen, Partner, Dependents
2. **Account Info:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
3. **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
4. **Target:** Churn (binary: Yes/No)

## Next Steps for Data Preparation
1. Clean TotalCharges column: convert whitespace to numeric, then handle appropriately
2. Convert SeniorCitizen to boolean or categorical as needed
3. Encode categorical variables for machine learning
4. Consider feature engineering (tenure buckets, service count, etc.)

## File Locations
- **Raw Dataset:** `data/raw/telco_churn.csv` (unchanged from original)
- **Data Dictionary:** `reports/data_dictionary.md`
- **Inspection Report:** `reports/dataset_inspection.md`

## Verification
✓ Dataset matches expected IBM Telco Customer Churn specifications
✓ 7,043 records, 21 columns
✓ Target variable distribution matches known statistics
✓ No modifications made to original raw dataset