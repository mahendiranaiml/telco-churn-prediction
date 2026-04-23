import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated
from zenml import step


def get_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.prediction")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger()


def validate_prediction_inputs(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    if model is None:
        raise ValueError("Trained model is required for prediction")
    if x_train.empty:
        raise ValueError("x_train is empty")
    if x_test.empty:
        raise ValueError("x_test is empty")
    if y_train.empty:
        raise ValueError("y_train is empty")
    if y_test.empty:
        raise ValueError("y_test is empty")


def predict_labels(model: Pipeline, x_test: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        threshold = getattr(model, "churn_threshold", 0.5)
        predictions = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)
        logger.info("Using prediction threshold %.2f.", threshold)
    else:
        predictions = model.predict(x_test)

    logger.info("Created predictions for %s test rows.", len(predictions))
    return pd.Series(predictions, index=x_test.index, name="prediction")


def predict_probabilities(model: Pipeline, x_test: pd.DataFrame) -> Optional[pd.Series]:
    if not hasattr(model, "predict_proba"):
        logger.warning("Model does not support predict_proba. Returning no probabilities.")
        return None

    probabilities = model.predict_proba(x_test)[:, 1]
    logger.info("Created prediction probabilities for %s test rows.", len(probabilities))
    return pd.Series(probabilities, index=x_test.index, name="prediction_probability")


def run_prediction(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Optional[pd.Series]]:
    validate_prediction_inputs(model, x_train, x_test, y_train, y_test)
    y_pred = predict_labels(model, x_test)
    y_pred_proba = predict_probabilities(model, x_test)
    return x_train, x_test, y_train, y_test, y_pred, y_pred_proba


@step
def predict_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_model: Pipeline,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[pd.Series, "y_pred"],
    Annotated[Optional[pd.Series], "y_pred_proba"],
]:
    x_train_output, x_test_output, y_train_output, y_test_output, y_pred, y_pred_proba = run_prediction(
        trained_model,
        x_train,
        x_test,
        y_train,
        y_test,
    )
    return x_train_output, x_test_output, y_train_output, y_test_output, y_pred, y_pred_proba
