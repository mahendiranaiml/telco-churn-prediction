# Telco Churn Prediction

End-to-end internship project for customer churn prediction:

`ingestion -> preprocessing -> training -> evaluation -> prediction -> app`

## Project Layout

```text
telco-churn-prediction/
├── app.py               # FastAPI app
├── config.yaml          # single project configuration file
├── index.html           # single-page web UI
├── data/                # raw, interim, processed datasets
├── models/              # saved model artifacts
├── notebooks/           # EDA notes and experiments
├── pipelines/           # simple pipeline wrappers
├── reports/             # metrics and plots
├── src/                 # direct Python files for ML steps
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── prediction.py
└── tests/               # basic tests
```

## Run

Train the model:

```bash
python -m src.training
```

Run the API:

```bash
uvicorn app:app --reload
```

Open `index.html` in your browser and submit a customer profile.
