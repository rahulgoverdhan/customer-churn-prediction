"""Clean the raw Telco Customer Churn dataset and save processed version."""
from pathlib import Path

import pandas as pd
import numpy as np

# Resolve paths relative to the repository root so the script runs from any directory
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / 'data' / 'raw' / 'telco_churn.csv'
PROCESSED_DATA_PATH = REPO_ROOT / 'data' / 'processed' / 'cleaned.csv'

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)

print('=== ORIGINAL DATASET ===')
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print()

# 1. Clean TotalCharges - convert whitespace to NaN
print('=== CLEANING TotalCharges ===')
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f'TotalCharges converted to numeric. NaN count: {df["TotalCharges"].isna().sum()}')
print(f'TotalCharges dtype: {df["TotalCharges"].dtype}')

# 2. Fix SeniorCitizen - convert to bool
print()
print('=== FIXING SeniorCitizen ===')
print(f'Original values: {df["SeniorCitizen"].unique()}')
df['SeniorCitizen'] = df['SeniorCitizen'].astype(bool)
print(f'Converted to bool: {df["SeniorCitizen"].unique()}')

# 3. Check for duplicate customerIDs
print()
print('=== CHECKING DUPLICATE IDs ===')
dup_ids = df['customerID'].duplicated().sum()
print(f'Duplicate customerIDs: {dup_ids}')

# 4. Check outliers using IQR method
print()
print('=== OUTLIER CHECK (IQR Method) ===')
for col in ['MonthlyCharges', 'TotalCharges', 'tenure']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f'{col}: {outliers} outliers (lower={lower:.2f}, upper={upper:.2f})')

# 5. Handle missing TotalCharges - impute with median
print()
print('=== HANDLING MISSING TotalCharges ===')
median_total = df['TotalCharges'].median()
print(f'Median TotalCharges: {median_total:.2f}')
missing_before = df['TotalCharges'].isna().sum()
df['TotalCharges'].fillna(median_total, inplace=True)
print(f'Filled {missing_before} missing values with median')

# 6. Verify final state
print()
print('=== FINAL DATASET ===')
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Duplicate rows: {df.duplicated().sum()}')
print()
print('Data types:')
print(df.dtypes)

# Save cleaned data
df.to_csv(PROCESSED_DATA_PATH, index=False)
print()
print(f'Cleaned dataset saved to {PROCESSED_DATA_PATH}')