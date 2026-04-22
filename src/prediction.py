import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline
from zenml import step


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.prediction")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()


class ModelPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        model: Pipeline,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series | None]:
        """Create predictions using a trained model."""


class TelcoChurnPredictor(ModelPredictor):
    def validate_inputs(
        self,
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

    def predict_test_data(self, model: Pipeline, x_test: pd.DataFrame) -> pd.Series:
        if hasattr(model, "predict_proba"):
            threshold = getattr(model, "churn_threshold", 0.5)
            predictions = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)
            logger.info("Using prediction threshold %.2f.", threshold)
        else:
            predictions = model.predict(x_test)
        logger.info("Created predictions for %s test rows.", len(predictions))
        return pd.Series(predictions, index=x_test.index, name="prediction")

    def predict_probabilities(self, model: Pipeline, x_test: pd.DataFrame) -> pd.Series | None:
        if not hasattr(model, "predict_proba"):
            logger.warning("Model does not support predict_proba. Returning no probabilities.")
            return None

        probabilities = model.predict_proba(x_test)[:, 1]
        logger.info("Created prediction probabilities for %s test rows.", len(probabilities))
        return pd.Series(probabilities, index=x_test.index, name="prediction_probability")

    def predict(
        self,
        model: Pipeline,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series | None]:
        self.validate_inputs(model, x_train, x_test, y_train, y_test)
        y_pred = self.predict_test_data(model, x_test)
        y_pred_proba = self.predict_probabilities(model, x_test)
        return x_train, x_test, y_train, y_test, y_pred, y_pred_proba


@step
def predict_model(
    x_train: pd.DataFrame, x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_model: Pipeline,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series | None]:

    predictor = TelcoChurnPredictor()
    predictor.validate_inputs(trained_model, x_train, x_test, y_train, y_test)
    y_pred = predictor.predict_test_data(trained_model, x_test)
    y_pred_proba = predictor.predict_probabilities(trained_model, x_test)
    return x_train, x_test, y_train, y_test, y_pred, y_pred_proba
