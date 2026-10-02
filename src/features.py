"""
Feature engineering module for customer churn prediction.
Creates meaningful features based on EDA insights.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union


class CustomerChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating customer churn prediction features.

    Features created:
    1. TenureGroup: Categorical tenure buckets (0-12, 13-24, 25-48, 49+ months)
    2. ServiceCount: Number of active services subscribed
    3. HasSecurityServices: Binary flag for security-related services
    4. AutoPay: Binary flag for automated payment methods
    5. MonthlyChargesPerService: Monthly charges per service (avoid div by zero)
    6. ContractOrdinal: Ordinal encoding of contract type
    7. HighValueCustomer: Binary flag for high monthly charges + long tenure
    8. TotalChargesPerMonth: Average monthly charges over tenure
    """

    def __init__(self):
        self.service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        self.security_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        self.auto_pay_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer (no fitting required for feature engineering).
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating new features.

        Args:
            X: Input DataFrame with raw features

        Returns:
            DataFrame with original + engineered features
        """
        if not self.is_fitted_:
            # Self-fit if not fitted
            self.fit(X)

        # Create a copy to avoid modifying original
        df = X.copy()

        # 1. Tenure Groups - Based on EDA showing different churn rates by tenure
        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=[-1, 12, 24, 48, 72],
            labels=['0-12 months', '13-24 months', '25-48 months', '49+ months'],
            ordered=True
        )

        # 2. Service Count - Number of services subscribed (EDA showed inverse relationship with churn)
        def count_services(row):
            count = 0
            for col in self.service_columns:
                # Handle special case: 'No phone service', 'No internet service' etc. count as 0
                val = row[col]
                if val == 'Yes':
                    count += 1
            return count

        df['ServiceCount'] = df.apply(count_services, axis=1)

        # 3. Has Security Services - Bundle of security-related services (EDA showed these reduce churn)
        def has_security_services(row):
            for col in self.security_services:
                if row[col] == 'Yes':
                    return 1
            return 0

        df['HasSecurityServices'] = df.apply(has_security_services, axis=1)

        # 4. Auto Pay - Automated payment methods (EDA showed lower churn)
        def is_auto_pay(row):
            return 1 if row['PaymentMethod'] in self.auto_pay_methods else 0

        df['AutoPay'] = df.apply(is_auto_pay, axis=1)

        # 5. Monthly Charges Per Service - Price efficiency metric
        # Add 1 to avoid division by zero for customers with 0 services
        df['MonthlyChargesPerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)

        # 6. Contract Ordinal - Ordinal encoding for contract (Month-to-month < One year < Two year)
        contract_mapping = {
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        }
        df['ContractOrdinal'] = df['Contract'].map(contract_mapping)

        # 7. High Value Customer - High monthly charges + long tenure (potential retention targets)
        df['HighValueCustomer'] = (
            (df['MonthlyCharges'] > 90) &
            (df['tenure'] > 12)
        ).astype(int)

        # 8. Total Charges Per Month - Average revenue per month
        # Handle tenure = 0 case (new customers)
        df['TotalChargesPerMonth'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']  # For new customers, use monthly charges as proxy
        )

        # 9. Avg Charge Per Service - Alternative to MonthlyChargesPerService
        df['AvgChargePerService'] = np.where(
            df['ServiceCount'] > 0,
            df['MonthlyCharges'] / df['ServiceCount'],
            df['MonthlyCharges']  # For 0 services, use monthly charges
        )

        return df

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        """
        if input_features is None:
            # Return all feature names including engineered ones
            # This is approximate - in practice, we'd track what we added
            return [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn',
                'TenureGroup', 'ServiceCount', 'HasSecurityServices', 'AutoPay',
                'MonthlyChargesPerService', 'ContractOrdinal', 'HighValueCustomer',
                'TotalChargesPerMonth', 'AvgChargePerService'
            ]
        else:
            # Return input features + our engineered features
            engineered_features = [
                'TenureGroup', 'ServiceCount', 'HasSecurityServices', 'AutoPay',
                'MonthlyChargesPerService', 'ContractOrdinal', 'HighValueCustomer',
                'TotalChargesPerMonth', 'AvgChargePerService'
            ]
            return list(input_features) + engineered_features


def create_preprocessing_pipeline():
    """
    Create a scikit-learn preprocessing pipeline for the churn prediction model.

    Returns:
        Pipeline: Scikit-learn preprocessing pipeline
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    from sklearn.pipeline import Pipeline

    # Define column types
    numeric_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'ServiceCount', 'MonthlyChargesPerService',
        'TotalChargesPerMonth', 'AvgChargePerService'
    ]

    # Binary features (0/1 already)
    binary_features = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'HasSecurityServices', 'AutoPay', 'HighValueCustomer',
        'PaperlessBilling'
    ]

    # Nominal categorical features
    nominal_features = [
        'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'PaymentMethod'
    ]

    # Ordinal categorical features
    ordinal_features = [
        'Contract',  # Will be handled specially
        'TenureGroup'  # Will be handled specially
    ]

    # Target
    target = ['Churn']

    # Preprocessing for numeric features: standardize
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for binary features: pass through (already 0/1)
    binary_transformer = Pipeline(steps=[
        ('pass', 'passthrough')
    ])

    # Preprocessing for nominal features: one-hot encode
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Preprocessing for ordinal features: ordinal encode
    # Note: Contract and TenureGroup have specific orderings
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder())
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ],
        remainder='drop'  # Drop any other columns (like customerID)
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    return pipeline, numeric_features, binary_features, nominal_features, ordinal_features


def prepare_modeling_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """
    Prepare data for modeling with train/validation/test split.

    Args:
        df: DataFrame with features and target
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split

    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})  # Convert to binary

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load cleaned data
    df = pd.read_csv('../data/processed/cleaned.csv')
    print(f"Original data shape: {df.shape}")

    # Apply feature engineering
    fe = CustomerChurnFeatureEngineer()
    df_engineered = fe.fit_transform(df)
    print(f"Engineered data shape: {df_engineered.shape}")
    print(f"New features: {[c for c in df_engineered.columns if c not in df.columns]}")

    # Show some engineered feature stats
    print("\\nTenure Group distribution:")
    print(df_engineered['TenureGroup'].value_counts())
    print("\\nService Count distribution:")
    print(df_engineered['ServiceCount'].value_counts().sort_index())
    print("\\nChurn rate by Tenure Group:")
    print(df_engineered.groupby('TenureGroup')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100))