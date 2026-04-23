# Telco Churn Prediction

This is an end-to-end machine learning project that I built for predicting customer churn in a telecom dataset. The goal is not only to train a model, but to show a complete practical workflow:

```text
data ingestion -> preprocessing -> training -> prediction -> evaluation -> FastAPI app -> Docker -> CI
```

I also added basic MLOps practices using ZenML for pipeline orchestration, MLflow for experiment tracking, Git/GitHub for version control, Docker for containerization, and GitHub Actions for basic CI.

## Problem Statement

Telecom companies lose customers when they churn. The goal of this project is to predict whether a customer is likely to churn based on demographic, contract, billing, and service-related information.

The target column is:

```text
Churn
```

Output:

```text
Yes -> customer is likely to churn
No  -> customer is not likely to churn
```

## Dataset Summary

The dataset contains:

```text
Total records: 7043
Not churned:   5174
Churned:       1869
```

Class distribution:

```text
No churn:  73.46%
Churn:     26.53%
```

This means the dataset is imbalanced. Because of this, accuracy alone is not enough. Recall is important because missing a real churn customer can be costly for the business.

## EDA Insights

During EDA, I found these important patterns:

`gender`

Churn rate was almost the same for male and female customers. This looked like a weak feature.

`Partner`

Customers without partners churned more often than customers with partners.

```text
Partner = No  -> higher churn
Partner = Yes -> lower churn
```

`tenure`

Tenure was one of the most important features. New customers had higher churn risk, while long-term customers were more stable.

`OnlineSecurity`

Customers without online security had much higher churn. This became one of the strongest service-related signals.

`Contract`

Month-to-month contract customers had higher churn risk. Longer contract customers were more stable.

`InternetService`

Fiber optic customers showed higher churn tendency, likely because of higher cost or service expectations.

`PaymentMethod`

Electronic check customers looked more churn-prone than other payment methods.

`MonthlyCharges`

Higher monthly charges were associated with higher churn risk.

## Feature Engineering

Based on the EDA, I improved preprocessing instead of only using raw columns.

I handled:

```text
customerID -> dropped because it is only an identifier
TotalCharges -> converted to numeric and missing values filled
No internet service -> simplified to No
No phone service -> simplified to No
duplicates -> removed
```

New features added:

```text
tenure_group
avg_monthly_spend
avg_charge
is_month_to_month
num_services
```

Why these features were added:

`tenure_group`

Captures customer lifecycle stages like new customers and long-term customers.

`avg_monthly_spend` and `avg_charge`

Help understand spending behavior more clearly than raw `TotalCharges` alone.

`is_month_to_month`

Marks customers with month-to-month contracts because this group is more churn-prone.

`num_services`

Counts how many optional services a customer uses. More services can indicate stronger customer attachment.

## Models Tried

I started with a baseline Random Forest model.

Then I tried multiple models:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

The pipeline compares these models on a validation split.

At first, optimizing mainly for F1 selected Gradient Boosting, but it reduced recall. That was not ideal for churn prediction because the business goal is to catch as many churn customers as possible.

So I changed the strategy.

## Final Modeling Strategy

The final strategy is:

```text
try multiple models
tune decision threshold
prioritize recall
keep validation F1 at a reasonable level
select final model
```

Threshold tuning checks thresholds from:

```text
0.25 to 0.60
```

This improved recall significantly.

Latest smoke-test result:

```text
Selected model: random_forest
Selected threshold: 0.25
Accuracy:  0.6548
Precision: 0.4282
Recall:    0.9059
F1:        0.5815
ROC AUC:   0.8434
```

Compared to the earlier baseline, recall improved from around:

```text
0.76 -> 0.91
```

Tradeoff:

Precision and accuracy dropped. This is expected because lowering the threshold makes the model more aggressive in predicting churn. For churn prediction, this can be acceptable because catching more churn-risk customers is often more important than being very conservative.

## Imbalance Handling

The dataset is imbalanced, so I handled it using:

```text
class_weight="balanced"
threshold tuning
recall-focused model selection
```

I did not use SMOTE yet. That can be a future experiment.

## MLOps Practices Used

`ZenML`

I used ZenML to orchestrate the ML pipeline.

Pipeline flow:

```text
ingest_data -> preprocess_data -> train_model -> predict_model -> evaluate_model
```

Run:

```bash
python -m pipelines.training_pipeline
```

Start ZenML UI:

```bash
zenml up
```

`MLflow`

I used MLflow to track:

```text
metrics
train rows
test rows
feature count
selected model
selected threshold
trained model artifact
```

Start MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

`config.yaml`

I used a single `config.yaml` file to keep project paths and settings in one place. This is better than hardcoding values everywhere.

`logger`

Each main step uses logging so I can understand what is happening during pipeline execution.

`Docker`

I built a Docker image for the FastAPI app.

Build:

```bash
docker build -t telco-churn-api .
```

Run:

```bash
docker run -p 8000:8000 telco-churn-api
```

`GitHub Actions`

I added basic CI using GitHub Actions.

Current CI does:

```text
checkout code
setup Python
install uv
install dependencies
check imports
build Docker image
```

This confirms that the project can set up and build on a fresh GitHub machine.

## API and UI

The project has a simple FastAPI backend and a small HTML/CSS/JS frontend.

Run the app:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Health check:

```text
http://127.0.0.1:8000/health
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
telco-churn-prediction/
|-- app.py
|-- config.yaml
|-- index.html
|-- Dockerfile
|-- pyproject.toml
|-- uv.lock
|-- data/
|-- models/
|-- notebooks/
|-- pipelines/
|   |-- training_pipeline.py
|   |-- inference_pipeline.py
|-- src/
|   |-- ingestion.py
|   |-- preprocessing.py
|   |-- training.py
|   |-- prediction.py
|   |-- evaluation.py
|-- reports/
|-- .github/workflows/
|   |-- ci.yml
```

## What I Struggled With and Solved

I faced several practical issues while building this project:

```text
ZenML dependency setup
FastAPI version conflicts
MLflow and pandas dependency conflict
Docker WSL integration
Docker build context issue because of .venv
GitHub Actions failing because pytest was not available
GitHub Actions failing when no tests existed
model recall vs precision tradeoff
threshold tuning tradeoff
```

How I solved them:

```text
fixed compatible package versions
used .dockerignore to avoid copying .venv
replaced pytest step with import check in CI
used class_weight and threshold tuning for imbalance
used MLflow to track experiments
used ZenML to organize the ML workflow
```

## Current Status

The project currently supports:

```text
EDA-based feature engineering
multiple model experiments
threshold tuning
recall-focused churn prediction
ZenML pipeline orchestration
MLflow tracking
FastAPI prediction app
Docker image build
GitHub Actions CI
```

## Future Improvements

Next improvements I would like to try:

```text
deeper EDA
feature importance analysis
SMOTE experiment
cross-validation
more threshold comparison
confusion matrix visualization
better frontend design
deployment to cloud
```

## Key Learning

This project helped me understand that machine learning projects are not only about training a model. A good project also needs:

```text
clean data pipeline
good feature engineering
experiment tracking
reproducible setup
API serving
containerization
CI checks
clear explanation of tradeoffs
```

For churn prediction, I learned that recall is very important because the goal is to identify as many churn-risk customers as possible.
