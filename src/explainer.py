"""
Stage 7: Model Explainability
Generate feature importance analysis, visualizations, and SHAP explanations
for the final tuned Logistic Regression model.
"""

import sys
from pathlib import Path
# Ensure src/ package is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8')

def get_feature_names(pipeline):
    """Extract all feature names from the pipeline."""
    preprocessor = pipeline.named_steps['preprocessor']
    if hasattr(preprocessor, 'get_feature_names_out'):
        return preprocessor.get_feature_names_out()
    else:
        return [f"feature_{i}" for i in range(len(pipeline.named_steps['classifier'].coef_[0]))]

def extract_coefficients(pipeline):
    """Extract coefficients and intercept from the model."""
    classifier = pipeline.named_steps['classifier']
    # For binary classification, coef_ is (1, n_features)
    coef = classifier.coef_[0]
    intercept = classifier.intercept_[0]
    return coef, intercept

def create_feature_importance_report(coef, feature_names, output_dir):
    """Create feature importance report and visualizations."""

    # Create DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'coefficient': coef,
        'absolute_coefficient': np.abs(coef)
    })

    # Sort by absolute coefficient (most important)
    feature_importance_df = feature_importance_df.sort_values('absolute_coefficient', ascending=False)

    # Create rankings
    feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)

    # Split into positive and negative predictors
    positive_predictors = feature_importance_df[feature_importance_df['coefficient'] > 0].copy()
    negative_predictors = feature_importance_df[feature_importance_df['coefficient'] < 0].copy()

    # Create visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Top 15 positive predictors
    top_pos = positive_predictors.head(15)

    plt.figure(figsize=(10, 8))
    # Sort for horizontal bar chart (most positive on top)
    top_pos = top_pos.sort_values('coefficient', ascending=True)

    colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in top_pos['coefficient']]
    bars = plt.barh(range(len(top_pos)), top_pos['coefficient'], color=colors, alpha=0.8)

    plt.yticks(range(len(top_pos)), top_pos['feature_name'])
    plt.xlabel('Coefficient Value (Positive = increases churn risk)')
    plt.title('Top 15 Positive Predictors of Customer Churn\n(Higher coefficient = higher churn probability)', fontsize=14)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if abs(width) > 0.001:
            plt.text(width + (0.01 if width > 0 else -0.02), i,
                    f'{width:.3f}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=9)

    plt.tight_layout()
    pos_plot_path = output_dir / 'top15_positive_predictors.png'
    plt.savefig(pos_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Top 15 negative predictors
    top_neg = negative_predictors.head(15)

    plt.figure(figsize=(10, 8))
    # Sort for horizontal bar chart (most negative on top)
    top_neg = top_neg.sort_values('coefficient', ascending=True)

    colors = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in top_neg['coefficient']]
    bars = plt.barh(range(len(top_neg)), top_neg['coefficient'], color=colors, alpha=0.8)

    plt.yticks(range(len(top_neg)), top_neg['feature_name'])
    plt.xlabel('Coefficient Value (Negative = decreases churn risk)')
    plt.title('Top 15 Negative Predictors of Customer Churn\n(Negative coefficient = reduces churn probability)', fontsize=14)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if abs(width) > 0.001:
            plt.text(width + (0.01 if width > 0 else -0.02), i,
                    f'{width:.3f}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=9)

    plt.tight_layout()
    neg_plot_path = output_dir / 'top15_negative_predictors.png'
    plt.savefig(neg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Coefficient distribution
    plt.figure(figsize=(12, 6))

    # Remove near-zero coefficients for better visualization
    non_zero_coef = feature_importance_df[np.abs(feature_importance_df['coefficient']) > 0.001]

    plt.hist(non_zero_coef['coefficient'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Logistic Regression Coefficients\n(Features with |coefficient| > 0.001)', fontsize=14)

    plt.tight_layout()
    dist_plot_path = output_dir / 'coefficient_distribution.png'
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save DataFrames
    top_pos.to_csv(output_dir / 'top15_positive_predictors.csv', index=False)
    top_neg.to_csv(output_dir / 'top15_negative_predictors.csv', index=False)
    feature_importance_df.to_csv(output_dir / 'all_feature_importance.csv', index=False)

    return {
        'positive_predictors': top_pos,
        'negative_predictors': top_neg,
        'all_features': feature_importance_df,
        'plots': {
            'positive_plot': str(pos_plot_path),
            'negative_plot': str(neg_plot_path),
            'distribution_plot': str(dist_plot_path)
        }
    }

def try_shap_explanations(pipeline, test_data, output_dir):
    """Attempt to generate SHAP explanations.

    Returns:
        dict: SHAP analysis results if successful, empty dict if failed
    """
    try:
        import shap

        output_dir = Path(output_dir)
        shap_dir = output_dir / 'shap'
        shap_dir.mkdir(exist_ok=True)

        # Generate SHAP explanations
        explainer = shap.Explainer(pipeline)
        shap_values = explainer(test_data)

        # Save summary plot
        shap.summary_plot(shap_values, test_data,
                         plot_type="bar",
                         show=False,
                         output_path=shap_dir / 'shap_summary_bar.png')

        # Save detailed force plot for first instance
        shap.force_plot(explainer.expected_value[0],
                       shap_values[0].values,
                       test_data.iloc[0],
                       matplotlib=False,
                       output_path=shap_dir / 'force_plot_instance0.html')

        # Save SHAP values
        np.save(shap_dir / 'shap_values.npy', shap_values.values)

        return {
            'success': True,
            'shap_dir': str(shap_dir),
            'explainer': explainer,
            'shap_values': shap_values
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'reason': 'SHAP not compatible with pipeline or missing dependencies'
        }

def create_business_report(positive_df, negative_df, final_roc_auc, output_path):
    """Create a business-meaningful report of churn factors."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CUSTOMER CHURN PREDICTION - MODEL EXPLAINABILITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Model performance summary
    report_lines.append("MODEL PERFORMANCE")
    report_lines.append("-" * 40)
    report_lines.append(f"Final Test ROC-AUC: {final_roc_auc:.4f}")
    report_lines.append("")

    # Top positive predictors
    report_lines.append("TOP FACTORS THAT INCREASE CHURN RISK (Positive Coefficients)")
    report_lines.append("-" * 70)

    for _, row in positive_df.head(10).iterrows():
        feature_name = row['feature_name']
        coefficient = row['coefficient']

        # Business interpretation based on feature name
        business_meaning = interpret_feature(feature_name, is_positive=True)
        report_lines.append(f"• {feature_name}: {coefficient:.3f}")
        report_lines.append(f"  Business Impact: {business_meaning}")
        report_lines.append("")

    # Top negative predictors
    report_lines.append("TOP FACTORS THAT REDUCE CHURN RISK (Negative Coefficients)")
    report_lines.append("-" * 70)

    for _, row in negative_df.head(10).iterrows():
        feature_name = row['feature_name']
        coefficient = row['coefficient']

        # Business interpretation based on feature name
        business_meaning = interpret_feature(feature_name, is_positive=False)
        report_lines.append(f"• {feature_name}: {coefficient:.3f}")
        report_lines.append(f"  Business Impact: {business_meaning}")
        report_lines.append("")

    # Key business insights
    report_lines.append("KEY BUSINESS INSIGHTS FOR CUSTOMER RETENTION")
    report_lines.append("-" * 50)
    report_lines.append("1. Focus retention efforts on customers with the highest churn risk.")
    report_lines.append("2. Leverage positive predictors to identify at-risk customer segments.")
    report_lines.append("3. Use negative predictors to understand what drives customer loyalty.")
    report_lines.append("4. Monitor key features regularly to detect churn patterns.")
    report_lines.append("")

    # Recommendation
    report_lines.append("PRIORITY ACTIONS")
    report_lines.append("-" * 20)
    report_lines.append("1. Target customers with high MonthlyCharges or no long-term contracts")
    report_lines.append("2. Implement retention offers for customers without auto-pay setup")
    report_lines.append("3. Focus on customers with fiber optic or DSL internet services")
    report_lines.append("4. Monitor customers with high TotalCharges and consider upgrades")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text

def interpret_feature(feature_name, is_positive):
    """Provide business interpretation of feature impact."""
    # Clean feature name
    clean_name = feature_name.replace('__', ' ')

    # Map technical feature names to business language
    interpretations = {
        'SeniorCitizen': 'Older customers may have different service needs',
        'tenure': 'Longer relationships with the company generally reduce churn',
        'MonthlyCharges': 'Higher monthly bills increase churn risk significantly',
        'TotalCharges': 'Cumulative spending correlates with customer lifetime value',
        'ServiceCount': 'Customers with more services are typically more engaged',
        'HasSecurityServices': 'Security services create stickiness and reduce churn',
        'AutoPay': 'Automated payments improve customer retention',
        'MonthlyChargesPerService': 'Price efficiency affects customer satisfaction',
        'ContractOrdinal': 'Long-term contracts (2-year) provide the strongest retention',
        'HighValueCustomer': 'High-value customers deserve special attention',
        'gender_Female': 'Female customers generally show different usage patterns',
        'Partner_Yes': 'Customers with partners may have different service needs',
        'Dependents_Yes': 'Family responsibilities influence service preferences',
        'PhoneService_Yes': 'Essential phone service retention is crucial',
        'InternetService': 'Type of internet service significantly impacts churn rates',
        'StreamingMovies': 'Entertainment services can be a retention anchor'
    }

    # Default interpretation
    if is_positive:
        return f"Feature '{clean_name}' positively contributes to churn risk"
    else:
        return f"Feature '{clean_name}' negatively contributes to churn risk (protective factor)"

def main():
    """Main explainability analysis function."""

    print("=" * 60)
    print("STAGE 7: MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    # Load the final model
    print("\nLoading final model from models/final_model.joblib...")
    model = joblib.load('models/final_model.joblib')
    print("✓ Model loaded successfully")

    # Get feature names
    print("\nExtracting feature names from pipeline...")
    feature_names = get_feature_names(model)
    print(f"✓ Extracted {len(feature_names)} feature names")

    # Extract coefficients
    print("\nExtracting model coefficients...")
    coef, intercept = extract_coefficients(model)
    print(f"✓ Extracted coefficients (intercept: {intercept:.4f})")

    # Create output directories
    reports_dir = Path("reports")
    models_dir = Path("models")
    explainability_dir = reports_dir / "explainability"
    explainability_dir.mkdir(parents=True, exist_ok=True)

    # Create feature importance analysis
    print("\nCreating feature importance analysis and visualizations...")
    feature_results = create_feature_importance_report(
        coef, feature_names, explainability_dir
    )
    print("✓ Feature importance analysis completed")

    # Try SHAP explanations
    print("\nAttempting SHAP explainability analysis...")
    # Create minimal test data for SHAP
    from sklearn.model_selection import train_test_split
    from src.model import load_and_prepare_data, create_data_splits

    X, y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    preprocessor = model.named_steps['preprocessor']

    X_test_transformed = preprocessor.transform(X_test.iloc[:100])  # Use first 100 for speed

    shap_results = try_shap_explanations(model, X_test_transformed, explainability_dir)

    if shap_results['success']:
        print("✓ SHAP analysis completed successfully")
        shap_info = {
            'success': True,
            'shap_dir': shap_results['shap_dir']
        }
    else:
        print(f"⚠ SHAP analysis not compatible: {shap_results['error']}")
        print("  Using coefficient-based analysis only")
        shap_info = {
            'success': False,
            'error': shap_results['error'],
            'reason': shap_results['reason']
        }

    # Create business report
    print("\nCreating business-meaningful report...")
    business_report_path = reports_dir / "churn_predictor_explainability_report.md"

    # Use the final ROC-AUC from our evaluation (0.8418)
    final_roc_auc = 0.8418

    report_text = create_business_report(
        feature_results['positive_predictors'],
        feature_results['negative_predictors'],
        final_roc_auc,
        business_report_path
    )
    print(f"✓ Business report saved to {business_report_path}")

    # Display key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FROM EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    print(f"\nFinal Test ROC-AUC: {final_roc_auc:.4f}")
    print(f"Total features analyzed: {len(feature_names)}")

    print("\nTop 5 Positive Churn Predictors:")
    for i, (_, row) in enumerate(feature_results['positive_predictors'].head().iterrows()):
        print(f"  {i+1}. {row['feature_name']}: {row['coefficient']:.4f}")

    print("\nTop 5 Negative Churn Predictors:")
    for i, (_, row) in enumerate(feature_results['negative_predictors'].head().iterrows()):
        print(f"  {i+1}. {row['feature_name']}: {row['coefficient']:.4f}")

    if shap_info['success']:
        print(f"\nSHAP Analysis: ✓ Successful")
        print(f"  SHAP output directory: {shap_info['shap_dir']}")
    else:
        print(f"\nSHAP Analysis: ⚠ Not compatible")
        print(f"  Reason: {shap_info.get('reason', 'Unknown')}")

    print(f"\nOutput files created:")
    for key, path in feature_results['plots'].items():
        print(f"  • {key}: {Path(path).name}")

    print(f"  • Data CSVs: top15_positive_predictors.csv, top15_negative_predictors.csv")
    print(f"  • Business report: {business_report_path.name}")
    if shap_info['success']:
        print(f"  • SHAP outputs: {Path(shap_info['shap_dir']).name}/")

    print("\n" + "=" * 60)
    print("STAGE 7 EXPLAINABILITY ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()