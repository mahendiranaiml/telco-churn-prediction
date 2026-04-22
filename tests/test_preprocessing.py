import pandas as pd

from src.preprocessing import TelcoDataPreprocessor


def test_cleaner_converts_total_charges():
    data = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cleaned = TelcoDataPreprocessor().clean_total_charges(data)
    assert cleaned["TotalCharges"].dtype.kind in "fi"


def test_feature_engineer_adds_features():
    data = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    transformed = TelcoDataPreprocessor().preprocess(data)
    assert "avg_monthly_spend" in transformed.columns
    assert "is_month_to_month" in transformed.columns
