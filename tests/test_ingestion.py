import pandas as pd

from src.ingestion import CSVDatasetIngestor


def test_validator_accepts_required_columns():
    data = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    CSVDatasetIngestor("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").validate_dataframe(data)
