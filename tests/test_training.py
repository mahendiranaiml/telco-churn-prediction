import pandas as pd

from src.training import TelcoChurnModelTrainer


def test_split_features_removes_target():
    data = pd.DataFrame({"feature": [1, 2], "Churn": ["No", "Yes"]})
    x, y = TelcoChurnModelTrainer().split_features_and_target(data)
    assert "Churn" not in x.columns
    assert y.tolist() == ["No", "Yes"]
