import logging
from abc import ABC, abstractmethod

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from zenml import step


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.evaluation")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()


class ModelEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | None,
    ) -> dict:
        """Evaluate model predictions."""


class TelcoChurnEvaluator(ModelEvaluator):
    def validate_inputs(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: pd.Series,
    ) -> None:
        if x_train.empty:
            raise ValueError("x_train is empty")

        if x_test.empty:
            raise ValueError("x_test is empty")

        if y_train.empty:
            raise ValueError("y_train is empty")

        if y_test.empty:
            raise ValueError("y_test is empty")

        if y_pred.empty:
            raise ValueError("y_pred is empty")

        if len(y_test) != len(y_pred):
            raise ValueError("y_test and y_pred must have the same length")

    def calculate_metrics(
        self,
        y_test: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | None,
    ) -> dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))

        logger.info("Evaluation metrics calculated: %s", metrics)
        return metrics

    def create_classification_report(self, y_test: pd.Series, y_pred: pd.Series) -> dict:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info("Classification report created.")
        return report

    def create_prediction_results(
        self,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | None,
    ) -> pd.DataFrame:
        results = x_test.copy()
        results["actual"] = y_test
        results["predicted"] = y_pred

        if y_pred_proba is not None:
            results["prediction_probability"] = y_pred_proba

        logger.info("Prediction results table created with %s rows.", results.shape[0])
        return results

    def log_to_mlflow(
        self,
        trained_model: Pipeline,
        metrics: dict[str, float],
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> None:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("telco_churn_prediction")
        with mlflow.start_run(run_name="telco_churn_model"):
            mlflow.log_metrics(metrics)
            mlflow.log_param("train_rows", x_train.shape[0])
            mlflow.log_param("test_rows", x_test.shape[0])
            mlflow.log_param("feature_count", x_train.shape[1])
            mlflow.sklearn.log_model(trained_model, "model")
        logger.info("Logged metrics and model to MLflow.")

    def evaluate(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series | None,
        trained_model: Pipeline | None = None,
    ) -> dict:
        self.validate_inputs(x_train, x_test, y_train, y_test, y_pred)
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        report = self.create_classification_report(y_test, y_pred)
        prediction_results = self.create_prediction_results(x_test, y_test, y_pred, y_pred_proba)

        if trained_model is not None:
            self.log_to_mlflow(trained_model, metrics, x_train, x_test)

        return {
            "metrics": metrics,
            "classification_report": report,
            "prediction_results": prediction_results,
            "train_rows": x_train.shape[0],
            "test_rows": x_test.shape[0],
            "feature_count": x_train.shape[1],
        }


@step
def evaluate_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_pred_proba: pd.Series | None,
    trained_model: Pipeline,
) -> dict:
    evaluator = TelcoChurnEvaluator()
    evaluator.validate_inputs(x_train, x_test, y_train, y_test, y_pred)
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    report = evaluator.create_classification_report(y_test, y_pred)
    prediction_results = evaluator.create_prediction_results(x_test, y_test, y_pred, y_pred_proba)
    evaluator.log_to_mlflow(trained_model, metrics, x_train, x_test)

    return {
        "metrics": metrics,
        "classification_report": report,
        "prediction_results": prediction_results,
        "train_rows": x_train.shape[0],
        "test_rows": x_test.shape[0],
        "feature_count": x_train.shape[1],
    }
