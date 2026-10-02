"""
Machine learning data preparation pipeline
for Customer Churn Prediction.
"""

import sys
from pathlib import Path
# Ensure src/ package is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.features import CustomerChurnFeatureEngineer


def load_and_prepare_data():
    """Load cleaned data and apply feature engineering."""

    df = pd.read_csv("data/processed/cleaned.csv")

    # Separate target from predictors
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Apply our custom feature engineering
    feature_engineer = CustomerChurnFeatureEngineer()
    X = feature_engineer.fit_transform(X, y)

    # Customer ID is an identifier, not a useful ML feature
    X = X.drop(columns=["customerID"], errors="ignore")

    return X, y


def create_preprocessor(X):
    """Create preprocessing pipeline for numerical and categorical features."""

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor, numerical_columns, categorical_columns


def create_data_splits(X, y):
    """Create stratified train, validation and test sets."""

    # First: 80% temporary training data, 20% test data
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Second: split remaining 80% into 60% train and 20% validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_temp,
        y_train_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_train_temp,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


if __name__ == "__main__":

    X, y = load_and_prepare_data()

    print("Feature engineering completed.")
    print("Final feature shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_data_splits(X, y)

    print("\nData splits:")
    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)

    preprocessor, numerical_columns, categorical_columns = (
        create_preprocessor(X_train)
    )

    print("\nPreprocessing configuration:")
    print("Numerical features:", len(numerical_columns))
    print("Categorical features:", len(categorical_columns))

    print("\nStage 5 data preparation completed successfully.")


# ============================================================
# STAGE 6: MODEL TRAINING AND VALIDATION
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


def train_and_evaluate_models():
    """Train baseline and tree-based models and compare validation performance."""

    X, y = load_and_prepare_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_data_splits(X, y)

    preprocessor, _, _ = create_preprocessor(X_train)

    # --------------------------------------------------------
    # Model 1: Logistic Regression
    # --------------------------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_pipeline.fit(X_train, y_train)

    logistic_predictions = logistic_pipeline.predict(X_validation)
    logistic_probabilities = logistic_pipeline.predict_proba(
        X_validation
    )[:, 1]

    logistic_results = {
        "Accuracy": accuracy_score(
            y_validation, logistic_predictions
        ),
        "ROC-AUC": roc_auc_score(
            y_validation, logistic_probabilities
        ),
        "Precision": precision_score(
            y_validation, logistic_predictions
        ),
        "Recall": recall_score(
            y_validation, logistic_predictions
        ),
        "F1": f1_score(
            y_validation, logistic_predictions
        ),
    }

    # --------------------------------------------------------
    # Model 2: Random Forest
    # --------------------------------------------------------

    # Create a fresh preprocessor for the second pipeline
    preprocessor_rf, _, _ = create_preprocessor(X_train)

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor_rf),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_pipeline.fit(X_train, y_train)

    rf_predictions = random_forest_pipeline.predict(X_validation)
    rf_probabilities = random_forest_pipeline.predict_proba(
        X_validation
    )[:, 1]

    rf_results = {
        "Accuracy": accuracy_score(
            y_validation, rf_predictions
        ),
        "ROC-AUC": roc_auc_score(
            y_validation, rf_probabilities
        ),
        "Precision": precision_score(
            y_validation, rf_predictions
        ),
        "Recall": recall_score(
            y_validation, rf_predictions
        ),
        "F1": f1_score(
            y_validation, rf_predictions
        ),
    }

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL VALIDATION RESULTS")
    print("=" * 60)

    print("\nLogistic Regression:")
    for metric, value in logistic_results.items():
        print(f"{metric:12}: {value:.4f}")

    print("\nRandom Forest:")
    for metric, value in rf_results.items():
        print(f"{metric:12}: {value:.4f}")

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_validation,
            logistic_predictions,
            target_names=["No Churn", "Churn"],
        )
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_validation,
            rf_predictions,
            target_names=["No Churn", "Churn"],
        )
    )

    return {
        "logistic": logistic_pipeline,
        "random_forest": random_forest_pipeline,
        "logistic_results": logistic_results,
        "random_forest_results": rf_results,
    }


# ============================================================
# STAGE 6B: LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ============================================================

def tune_logistic_regression():
    """Tune Logistic Regression using cross-validation."""

    X, y = load_and_prepare_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_data_splits(X, y)

    preprocessor, _, _ = create_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 60)

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"Best cross-validation ROC-AUC: "
        f"{grid_search.best_score_:.4f}"
    )

    validation_probabilities = grid_search.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = grid_search.predict(
        X_validation
    )

    print("\nValidation Results:")

    print(
        f"ROC-AUC   : "
        f"{roc_auc_score(y_validation, validation_probabilities):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_validation, validation_predictions):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_validation, validation_predictions):.4f}"
    )

    print(
        f"F1        : "
        f"{f1_score(y_validation, validation_predictions):.4f}"
    )

    return grid_search


# ============================================================
# STAGE 6C: FINAL MODEL EVALUATION ON TEST SET
# ============================================================

import joblib
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_final_model():
    """Evaluate the final tuned Logistic Regression model on the untouched test set.

    Uses the best-tuned C=0.1 Logistic Regression pipeline:
      - Fit on the training split only
      - Evaluate once on the held-out test split
      - Report accuracy, ROC-AUC, precision, recall, F1, confusion matrix, classification report
      - Save the trained pipeline to models/ after successful evaluation
    """

    X, y = load_and_prepare_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = create_data_splits(X, y)

    preprocessor, _, _ = create_preprocessor(X_train)

    # --------------------------------------------------------
    # Final model: Logistic Regression with C=0.1 (tuned)
    # --------------------------------------------------------

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    print("\nFitting final pipeline on training data...")
    final_pipeline.fit(X_train, y_train)

    # Predictions on the untouched test set
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Save the final trained pipeline to models/
    joblib.dump(final_pipeline, "models/final_model.joblib")
    print("\nFinal trained pipeline saved to models/final_model.joblib")

    return final_pipeline


if __name__ == "__main__":
    evaluate_final_model()