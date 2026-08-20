# Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecommunications customer is likely to churn based on customer demographics, services, contract details, and billing information.

## Project Overview

Customer churn is an important business problem for telecommunications companies. This project uses the IBM Telco Customer Churn dataset to build and compare multiple classification models for predicting customer churn.

The project follows a complete machine learning workflow:

- Data loading and exploration
- Data cleaning
- Missing-value handling
- Feature and target separation
- Train-test splitting
- Categorical feature encoding
- Numerical feature scaling
- Model training
- Model comparison
- Model evaluation
- Logistic Regression coefficient-based explainability

## Dataset

**Dataset:** IBM Telco Customer Churn

The original dataset contains 7,043 customer records and 21 columns.

During preprocessing:

- 11 rows with missing `TotalCharges` values were removed.
- The final dataset contains 7,032 records.
- `customerID` was removed because it is an identifier rather than a predictive feature.
- The final feature set contains 19 predictive features.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab
- Jupyter Notebook
- GitHub

## Machine Learning Workflow

### 1. Data Preprocessing

The dataset contains both numerical and categorical variables.

**Numerical features:**
- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

**Categorical features:**
- gender
- Partner
- Dependents
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod

Categorical features were transformed using One-Hot Encoding, while numerical features were standardized using StandardScaler.

### 2. Train-Test Split

The data was divided using an 80/20 stratified train-test split.

- Training records: 5,625
- Testing records: 1,407

Stratification was used to preserve the churn class distribution.

## Models Compared

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Decision Tree | 78.96% | 60.21% | 61.50% | 60.85% | 82.96% |
| Random Forest | 79.10% | 63.07% | 51.60% | 56.76% | 83.02% |

## Final Model

**Logistic Regression** was selected as the final model based on its overall performance.

Final test-set results:

- Accuracy: **80.38%**
- Precision: **64.85%**
- Recall: **57.22%**
- F1-Score: **60.80%**
- ROC-AUC: **83.59%**

A Scikit-learn Pipeline was created to combine preprocessing and the final Logistic Regression model.

## Model Explainability

Logistic Regression coefficients were analyzed to understand which features were associated with higher or lower predicted churn probability.

### Strongest Positive Coefficients

- `TotalCharges`: 0.6440
- `Contract_Month-to-month`: 0.6138
- `InternetService_Fiber optic`: 0.5902
- `StreamingTV_Yes`: 0.1911
- `PaymentMethod_Electronic check`: 0.1807

### Strongest Negative Coefficients

- `tenure`: -1.3523
- `Contract_Two year`: -0.7790
- `InternetService_DSL`: -0.6161
- `MonthlyCharges`: -0.5410
- `PaperlessBilling_No`: -0.3004

These coefficients represent associations learned by the model and should not be interpreted as proof of causation.

