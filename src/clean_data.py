"""Clean the raw Telco Customer Churn dataset and save processed version."""
import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv('C:/Users/omen/customer-churn-predict/data/raw/telco_churn.csv')

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
df.to_csv('C:/Users/omen/customer-churn-predict/data/processed/cleaned.csv', index=False)
print()
print('Cleaned dataset saved to data/processed/cleaned.csv')