# EDA Findings Report - Customer Churn Prediction

## Executive Summary
This report summarizes the comprehensive Exploratory Data Analysis (EDA) performed on the cleaned Telco Customer Churn dataset (7,043 customers, 21 features). The analysis identifies key drivers of customer churn and provides actionable business insights for retention strategies.

## Dataset Overview
- **Total Records:** 7,043
- **Features:** 21 (1 identifier + 20 features)
- **Target Variable:** Churn (Yes/No)
- **Churn Rate:** 26.5% (1,869 churned / 5,174 retained)
- **Data Quality:** 0 missing values, 0 duplicate rows

---

## Key Findings by Category

### 1. Contract Type (Strongest Predictor)
| Contract Type | Churn Rate | Customers |
|---------------|------------|-----------|
| Month-to-month | **42.7%** | 3,875 |
| One year | 11.3% | 1,473 |
| Two year | 2.8% | 1,695 |

**Insight:** Month-to-month customers are 15x more likely to churn than two-year contract customers. Contract type is the single strongest predictor of churn.

---

### 2. Internet Service
| Internet Service | Churn Rate | Customers |
|------------------|------------|-----------|
| Fiber optic | **41.9%** | 3,096 |
| DSL | 19.0% | 2,421 |
| No internet | 7.4% | 1,526 |

**Insight:** Fiber optic customers have the highest churn rate despite higher monthly charges. This suggests service quality or competitive pricing issues.

---

### 3. Payment Method
| Payment Method | Churn Rate | Customers |
|----------------|------------|-----------|
| Electronic check | **45.3%** | 2,365 |
| Mailed check | 19.1% | 1,612 |
| Bank transfer (automatic) | 16.7% | 1,544 |
| Credit card (automatic) | 15.2% | 1,522 |

**Insight:** Electronic check users have the highest churn (45.3%). Automated payment methods (bank transfer, credit card) show significantly lower churn.

---

### 4. Tenure Analysis
| Tenure Group | Churn Rate | Key Insight |
|--------------|------------|-------------|
| 0-12 months | **57.3%** | Highest vulnerability period |
| 13-24 months | 36.2% | Still high risk |
| 25-48 months | 24.2% | Moderate risk |
| 49+ months | **9.0%** | Lowest risk - loyal customers |

**Insight:** First-year customers are the most vulnerable. Early retention programs targeting new customers could have the highest ROI.

---

### 5. Monthly Charges Distribution
- **High charges (>$90/month):** Significantly higher churn
- **Low charges (<$50/month):** Lower churn
- **Median:** $70.35/month
- **Churned customers median:** Higher than retained customers

**Insight:** Price sensitivity is a major driver. Competitive pricing or loyalty discounts could reduce churn.

---

### 6. Service Count (New Feature)
| Services | Churn Rate | Customers |
|----------|------------|-----------|
| 0-2 services | ~26-30% | Higher churn |
| 3+ services | ~15% | Lower churn |

**Insight:** Customers with more bundled services are less likely to churn. Cross-selling and bundling is an effective retention strategy.

---

### 7. Demographic Factors
| Factor | Higher Churn Group | Churn Rate |
|--------|-------------------|------------|
| Senior Citizen | Seniors | 41.7% vs 23.6% |
| Paperless Billing | Yes | 33.6% vs 16.3% |
| Gender | No significant difference | ~26.5% both |
| Partner | No | 34.6% vs 20.4% |
| Dependents | No | 32.2% vs 22.9% |

**Insight:** Senior citizens and customers without partners/dependents churn more. Paperless billing customers surprisingly churn more (possible correlation with tech-savviness or automated billing issues).

---

### 8. Service-Specific Features (New Analyses)
| Service | Highest Churn Category | Rate | Lowest Churn Category | Rate |
|---------|------------------------|------|----------------------|------|
| PhoneService | No phone service | ~30% | Yes | ~23% |
| MultipleLines | No phone service | ~30% | Yes | ~18% |
| OnlineSecurity | No | 41.8% | Yes | 15.8% |
| OnlineBackup | No | 42.7% | Yes | 17.4% |
| DeviceProtection | No | 40.8% | Yes | 17.2% |
| TechSupport | No | 41.2% | Yes | 15.9% |
| StreamingTV | No | 38.9% | Yes | 18.2% |
| StreamingMovies | No | 38.5% | Yes | 17.9% |

**Insight:** Customers **without** add-on services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport) have 2.5-2.7x higher churn rates. These services appear to increase stickiness.

---

### 9. Correlation Analysis (Numerical Features)
| Feature Pair | Correlation | Interpretation |
|--------------|-------------|----------------|
| Tenure ↔ TotalCharges | **0.83** | Strong positive - longer tenure = more total charges |
| MonthlyCharges ↔ TotalCharges | 0.65 | Moderate positive |
| SeniorCitizen ↔ Tenure | -0.15 | Seniors tend to have lower tenure |
| Tenure ↔ MonthlyCharges | -0.01 | No correlation |

**Insight:** Tenure and TotalCharges are highly correlated (expected). No strong correlation between tenure and monthly charges.

---

## Priority Retention Strategies (Ranked by Impact)

1. **MIGRATE MONTH-TO-MONTH CUSTOMERS TO LONG-TERM CONTRACTS**
   - Target: 3,875 customers (42.7% churn rate)
   - Potential impact: Could reduce churn by ~40% for this segment

2. **ADDRESS FIBER OPTIC CHURN CRISIS**
   - Target: 3,096 customers (41.9% churn rate)
   - Investigate: Service quality, pricing, competitive offers

3. **TARGET ELECTRONIC CHECK USERS**
   - Target: 2,365 customers (45.3% churn rate)
   - Action: Incentivize migration to auto-pay (bank transfer/credit card)

4. **IMPLEMENT EARLY RETENTION PROGRAM (0-12 MONTHS)**
   - Target: ~1,800 customers in first year
   - Action: Onboarding program, proactive support, early loyalty rewards

5. **PROMOTE SERVICE BUNDLES**
   - Target: Customers with 1-2 services
   - Action: Bundle OnlineSecurity + DeviceProtection + TechSupport

6. **SENIOR CUSTOMER RETENTION PROGRAM**
   - Target: ~1,100 senior customers (41.7% churn rate)
   - Action: Dedicated support line, simplified billing, loyalty discounts

7. **PRICE SENSITIVITY MITIGATION**
   - Target: High monthly charge customers (>$90)
   - Action: Loyalty pricing, competitive match guarantees

---

## Feature Engineering Recommendations

Based on EDA findings, the following engineered features are recommended:

1. **TenureGroup** (0-12, 13-24, 25-48, 49+ months) - Strong predictive power
2. **ServiceCount** (0-9) - Clear inverse relationship with churn
3. **HasSecurityServices** (OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport) - Binary flag
4. **AutoPay** (Bank transfer + Credit card) - Binary flag for automated payments
5. **MonthlyChargesPerService** (MonthlyCharges / ServiceCount) - Price efficiency metric
6. **ContractOrdinal** (Month-to-month=1, One year=2, Two year=3) - Ordinal encoding
7. **HighValueCustomer** (MonthlyCharges > 90 AND Tenure > 12) - Binary flag

---

## Model Preparation Notes

### Class Imbalance
- Churn class: 26.5% (minority)
- Non-churn class: 73.5% (majority)
- Recommendation: Use class weights, SMOTE, or stratified sampling during training

### Categorical Encoding Required
- Nominal: gender, PaymentMethod, InternetService, Contract, MultipleLines, etc.
- Ordinal: Contract (Month-to-month < One year < Two year)
- Binary: SeniorCitizen, Partner, Dependents, PhoneService, PaperlessBilling

### Potential Leakage Risk
- **TotalCharges** is highly correlated with Tenure (0.83) - consider using only one
- **customerID** should be excluded from training

---

## Visualizations Generated
1. Churn Distribution (count plot)
2. Churn Rate by Contract Type (bar chart)
3. Tenure vs Churn (box plot)
4. Monthly Charges vs Churn (box plot)
5. Churn Rate by Internet Service (bar chart)
6. Correlation Heatmap (numerical features)
7. Churn Rate by Payment Method (bar chart)
8. Churn Rate by Paperless Billing (bar chart)
9. Churn Rate by Senior Citizen Status (bar chart)
10. Churn Rate by All Remaining Categorical Features (3x3 subplot grid)
11. Churn Rate by Tenure Group (bar chart)
12. Churn Rate by Service Count (grouped bar chart)

---

## Next Steps
1. Feature Engineering (`src/features.py`)
2. Train/Validation/Test Split
3. Baseline Model Training (`src/model.py`)
4. Model Evaluation & Hyperparameter Tuning
5. SHAP Interpretability Analysis (`src/interpret.py`)
6. Streamlit Dashboard Development (`src/app.py`)