from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import TelcoDataPreprocessor


MODEL_PATH = Path("models/churn_model.pkl")


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def prepare_input(payload: dict) -> pd.DataFrame:
    prepared_payload = TelcoDataPreprocessor().preprocess_payload(payload)
    return pd.DataFrame([prepared_payload])


def make_prediction(model, input_data: pd.DataFrame) -> dict:
    probability = float(model.predict_proba(input_data)[:, 1][0])
    threshold = getattr(model, "churn_threshold", 0.5)
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "Yes" if prediction == 1 else "No",
        "churn_probability": probability,
        "threshold": threshold,
    }


def inference_pipeline(payload: dict) -> dict:
    model = load_model()
    input_data = prepare_input(payload)
    return make_prediction(model, input_data)
