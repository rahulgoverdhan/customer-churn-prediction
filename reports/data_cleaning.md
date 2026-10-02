# Data Cleaning Report

## Overview
Cleaning transformations applied to the raw Telco Customer Churn dataset.

## Transformations Applied

### 1. TotalCharges Column
**Issue:** 11 records contained whitespace instead of numeric values (all corresponded to customers with tenure = 0).
**Fix:** Replaced whitespace with NaN, converted column to float64, then imputed NaN values with median (1397.47).
**Result:** TotalCharges now stored as float64 with 0 missing values.

### 2. SeniorCitizen Column
**Issue:** Stored as int64 (0/1).
**Fix:** Converted to bool (False/True).
**Result:** SeniorCitizen now stored as bool for semantic clarity.

### 3. Duplicate Records
**Finding:** 0 duplicate rows found. No action needed.

### 4. Outlier Analysis (IQR Method)
| Column | Outlier Count | Lower Bound | Upper Bound |
|--------|---------------|-------------|-------------|
| MonthlyCharges | 0 | -46.02 | 171.38 |
| TotalCharges | 0 | -4688.48 | 8884.67 |
| tenure | 0 | -60.00 | 124.00 |

**Finding:** No extreme outliers detected. All values fall within expected ranges for telecom customer data. No removal needed.

## Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Rows | 7,043 |
| Columns | 21 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Churn rate | 26.5% (1,869 Yes / 5,174 No) |

## Data Types After Cleaning

| Column | Type Before | Type After |
|--------|-------------|------------|
| TotalCharges | object (string) | float64 |
| SeniorCitizen | int64 | bool |
| All others | unchanged | unchanged |

## Files
- **Raw dataset:** `data/raw/telco_churn.csv` (unchanged)
- **Cleaned dataset:** `data/processed/cleaned.csv`
- **Cleaning script:** `src/clean_data.py`

## Notes
- The original raw dataset was not modified.
- Median imputation for TotalCharges was chosen over row deletion to preserve all 7,043 records.
- No scaling was applied - scaling is handled in the feature engineering pipeline for ML.