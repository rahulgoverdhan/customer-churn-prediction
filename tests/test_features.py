import pandas as pd

from src.features import CustomerChurnFeatureEngineer


def sample_customers():
    return pd.DataFrame({
        "tenure": [6, 18, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "PaymentMethod": [
            "Electronic check",
            "Bank transfer (automatic)",
            "Mailed check",
        ],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "MonthlyCharges": [50.0, 100.0, 80.0],
        "TotalCharges": [300.0, 1800.0, 4800.0],
    })


def test_feature_engineer_preserves_input():
    data = sample_customers()
    original = data.copy(deep=True)

    result = CustomerChurnFeatureEngineer().fit_transform(data)

    pd.testing.assert_frame_equal(data, original)
    assert len(result) == len(data)


def test_expected_features_are_created():
    result = CustomerChurnFeatureEngineer().fit_transform(
        sample_customers()
    )

    expected = {
        "TenureGroup",
        "ServiceCount",
        "HasSecurityServices",
        "AutoPay",
        "MonthlyChargesPerService",
        "ContractOrdinal",
        "HighValueCustomer",
        "TotalChargesPerMonth",
        "AvgChargePerService",
    }

    assert expected.issubset(result.columns)


def test_zero_service_and_zero_tenure_are_handled():
    data = sample_customers()
    data.loc[2, "tenure"] = 0
    data.loc[2, "TotalCharges"] = 0.0

    result = CustomerChurnFeatureEngineer().fit_transform(data)

    assert result.loc[2, "TotalChargesPerMonth"] == (
        result.loc[2, "MonthlyCharges"]
    )
    assert result.loc[2, "AvgChargePerService"] == (
        result.loc[2, "MonthlyCharges"]
    )