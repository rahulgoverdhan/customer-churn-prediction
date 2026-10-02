"""Verification script to test notebook logic."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/processed/cleaned.csv')
print(f"Data loaded: {df.shape}")

# Test TenureGroup creation
df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72],
                           labels=['0-12 months', '13-24 months', '25-48 months', '49+ months'])
print("TenureGroup created successfully")

# Test ServiceCount creation
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

def count_services(row):
    count = 0
    for col in service_cols:
        val = row[col]
        if val == 'Yes':
            count += 1
    return count

df['ServiceCount'] = df.apply(count_services, axis=1)
print("ServiceCount created successfully")

# Test categorical analysis
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']

for col in categorical_cols:
    churn_rate = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
    print(f"{col} vs Churn: OK")

# Test tenure churn
tenure_churn = pd.crosstab(df['TenureGroup'], df['Churn'], normalize='index') * 100
print("Tenure churn analysis: OK")

# Test service churn
service_churn = pd.crosstab(df['ServiceCount'], df['Churn'], normalize='index') * 100
print("Service churn analysis: OK")

# Test plotting
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(categorical_cols[:9]):
    if idx < len(axes):
        churn_rate = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        churn_rate['Yes'].sort_values(ascending=False).plot(kind='bar', ax=axes[idx], color='#ff6b6b')
        axes[idx].set_title(f'{col} vs Churn')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Churn Rate (%)')
        axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.close()
print("Categorical plots: OK")

# Test service count plot
grouped = df.groupby(['ServiceCount', 'Churn']).size().reset_index(name='count')
grouped['churn_rate'] = grouped.groupby('ServiceCount')['count'].transform(lambda x: x / x.sum() * 100)

pivot = grouped.pivot(index='ServiceCount', columns='Churn', values='churn_rate')
plt.figure(figsize=(10, 5))
pivot['Yes'].plot(kind='bar', color='#ff6b6b', alpha=0.7)
pivot['No'].plot(kind='bar', color='#4ecdc4', alpha=0.7)
plt.title("Churn Rate by Number of Services Subscribed")
plt.xlabel("Number of Services")
plt.ylabel("Churn Rate (%)")
plt.legend(title='Churn')
plt.close()
print("Service count plot: OK")

# Test tenure bucket plot
plt.figure(figsize=(8, 5))
tenure_churn['Yes'].sort_values(ascending=False).plot(kind='bar', color='#ff6b6b')
plt.title("Churn Rate by Tenure Group")
plt.xlabel("Tenure Group (Months)")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.close()
print("Tenure bucket plot: OK")

print("\n" + "="*60)
print("ALL NOTEBOOK CELLS VERIFIED SUCCESSFULLY!")
print("="*60)