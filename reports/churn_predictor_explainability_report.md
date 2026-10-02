================================================================================
CUSTOMER CHURN PREDICTION - MODEL EXPLAINABILITY REPORT
================================================================================
Generated: 2026-09-30 22:18:14

MODEL PERFORMANCE
----------------------------------------
Final Test ROC-AUC: 0.8418

TOP FACTORS THAT INCREASE CHURN RISK (Positive Coefficients)
----------------------------------------------------------------------
• cat__Dependents_No: 0.097
  Business Impact: Feature 'cat Dependents_No' positively contributes to churn risk

• cat__StreamingMovies_Yes: 0.116
  Business Impact: Feature 'cat StreamingMovies_Yes' positively contributes to churn risk

• num__MonthlyCharges: 0.141
  Business Impact: Feature 'num MonthlyCharges' positively contributes to churn risk

• cat__PaperlessBilling_Yes: 0.145
  Business Impact: Feature 'cat PaperlessBilling_Yes' positively contributes to churn risk

• cat__StreamingTV_Yes: 0.149
  Business Impact: Feature 'cat StreamingTV_Yes' positively contributes to churn risk

• cat__TenureGroup_0-12 months: 0.157
  Business Impact: Feature 'cat TenureGroup_0-12 months' positively contributes to churn risk

• cat__PaymentMethod_Electronic check: 0.157
  Business Impact: Feature 'cat PaymentMethod_Electronic check' positively contributes to churn risk

• cat__MultipleLines_No phone service: 0.174
  Business Impact: Feature 'cat MultipleLines_No phone service' positively contributes to churn risk

• cat__PhoneService_No: 0.174
  Business Impact: Feature 'cat PhoneService_No' positively contributes to churn risk

• cat__TechSupport_No: 0.184
  Business Impact: Feature 'cat TechSupport_No' positively contributes to churn risk

TOP FACTORS THAT REDUCE CHURN RISK (Negative Coefficients)
----------------------------------------------------------------------
• num__tenure: -0.782
  Business Impact: Feature 'num tenure' negatively contributes to churn risk (protective factor)

• num__ContractOrdinal: -0.547
  Business Impact: Feature 'num ContractOrdinal' negatively contributes to churn risk (protective factor)

• cat__InternetService_DSL: -0.291
  Business Impact: Feature 'cat InternetService_DSL' negatively contributes to churn risk (protective factor)

• cat__Contract_Two year: -0.245
  Business Impact: Feature 'cat Contract_Two year' negatively contributes to churn risk (protective factor)

• cat__TenureGroup_13-24 months: -0.237
  Business Impact: Feature 'cat TenureGroup_13-24 months' negatively contributes to churn risk (protective factor)

• cat__MultipleLines_No: -0.221
  Business Impact: Feature 'cat MultipleLines_No' negatively contributes to churn risk (protective factor)

• cat__PhoneService_Yes: -0.178
  Business Impact: Feature 'cat PhoneService_Yes' negatively contributes to churn risk (protective factor)

• num__HasSecurityServices: -0.159
  Business Impact: Feature 'num HasSecurityServices' negatively contributes to churn risk (protective factor)

• cat__PaperlessBilling_No: -0.149
  Business Impact: Feature 'cat PaperlessBilling_No' negatively contributes to churn risk (protective factor)

• cat__TenureGroup_25-48 months: -0.138
  Business Impact: Feature 'cat TenureGroup_25-48 months' negatively contributes to churn risk (protective factor)

KEY BUSINESS INSIGHTS FOR CUSTOMER RETENTION
--------------------------------------------------
1. Focus retention efforts on customers with the highest churn risk.
2. Leverage positive predictors to identify at-risk customer segments.
3. Use negative predictors to understand what drives customer loyalty.
4. Monitor key features regularly to detect churn patterns.

PRIORITY ACTIONS
--------------------
1. Target customers with high MonthlyCharges or no long-term contracts
2. Implement retention offers for customers without auto-pay setup
3. Focus on customers with fiber optic or DSL internet services
4. Monitor customers with high TotalCharges and consider upgrades

================================================================================
END OF REPORT
================================================================================