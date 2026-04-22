import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from zenml import step

from src.preprocessing import TARGET_COLUMN, TelcoDataPreprocessor


DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
THRESHOLDS = np.arange(0.25, 0.61, 0.05)
MIN_VALIDATION_F1 = 0.60


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.training")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()


class ModelTrainer(ABC):
    @abstractmethod
    def train(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
        """Train a model using processed data."""


class TelcoChurnModelTrainer(ModelTrainer):
    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        test_size: float = DEFAULT_TEST_SIZE,
        validation_size: float = DEFAULT_VALIDATION_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.candidate_scores: dict[str, dict[str, float]] = {}

    def validate_training_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Training data is empty")

        if self.target_column not in data.columns:
            raise ValueError(f"Target column not found: {self.target_column}")

        if data[self.target_column].nunique() < 2:
            raise ValueError("Target column must contain at least two classes")

    def split_features_and_target(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        logger.info("Created feature matrix with %s columns.", x.shape[1])
        return x, y

    def encode_target(self, y: pd.Series) -> pd.Series:
        encoded_target = self.label_encoder.fit_transform(y)
        logger.info("Encoded target classes: %s", list(self.label_encoder.classes_))
        return pd.Series(encoded_target, index=y.index, name=y.name)

    def split_train_test(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info("Train shape: %s, test shape: %s", x_train.shape, x_test.shape)
        return x_train, x_test, y_train, y_test

    def build_candidate_models(self) -> dict[str, object]:
        return {
            "logistic_regression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
        }

    def tune_threshold(self, probabilities, y_true: pd.Series) -> tuple[float, dict[str, float]]:
        best_threshold = 0.5
        best_metrics = {"recall": 0.0, "precision": 0.0, "f1": 0.0}
        best_recall_score = (-1.0, -1.0)
        best_f1_score = (-1.0, -1.0)
        best_f1_threshold = 0.5
        best_f1_metrics = best_metrics.copy()

        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)
            recall = float(recall_score(y_true, predictions, zero_division=0))
            precision = float(precision_score(y_true, predictions, zero_division=0))
            f1 = float(f1_score(y_true, predictions, zero_division=0))

            if (f1, recall) > best_f1_score:
                best_f1_threshold = float(threshold)
                best_f1_metrics = {"recall": recall, "precision": precision, "f1": f1}
                best_f1_score = (f1, recall)

            if f1 >= MIN_VALIDATION_F1 and (recall, f1) > best_recall_score:
                best_threshold = float(threshold)
                best_metrics = {"recall": recall, "precision": precision, "f1": f1}
                best_recall_score = (recall, f1)

        if best_recall_score[0] < 0:
            return best_f1_threshold, best_f1_metrics
        return best_threshold, best_metrics

    def build_pipeline(self, data: pd.DataFrame, model: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessor", TelcoDataPreprocessor().build_column_transformer(data)),
                ("model", model),
            ]
        )

    def choose_best_model(self, x_train: pd.DataFrame, y_train: pd.Series, data: pd.DataFrame) -> tuple[str, object, float]:
        train_x, valid_x, train_y, valid_y = train_test_split(
            x_train,
            y_train,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y_train,
        )

        best_name = ""
        best_model = None
        best_threshold = 0.5
        best_score = (-1.0, -1.0)

        for name, model in self.build_candidate_models().items():
            pipeline = self.build_pipeline(data, model)
            pipeline.fit(train_x, train_y)

            if hasattr(pipeline, "predict_proba"):
                probabilities = pipeline.predict_proba(valid_x)[:, 1]
                threshold, metrics = self.tune_threshold(probabilities, valid_y)
            else:
                predictions = pipeline.predict(valid_x)
                threshold = 0.5
                metrics = {
                    "recall": float(recall_score(valid_y, predictions, zero_division=0)),
                    "precision": float(precision_score(valid_y, predictions, zero_division=0)),
                    "f1": float(f1_score(valid_y, predictions, zero_division=0)),
                }

            self.candidate_scores[name] = {"threshold": threshold, **metrics}
            logger.info(
                "%s validation threshold=%.2f recall=%.4f precision=%.4f f1=%.4f",
                name,
                threshold,
                metrics["recall"],
                metrics["precision"],
                metrics["f1"],
            )

            if (metrics["recall"], metrics["f1"]) > best_score:
                best_name = name
                best_model = model
                best_threshold = threshold
                best_score = (metrics["recall"], metrics["f1"])

        logger.info("Selected final model: %s with threshold %.2f", best_name, best_threshold)
        return best_name, best_model, best_threshold

    def fit_final_model(self, data: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        selected_name, selected_model, selected_threshold = self.choose_best_model(x_train, y_train, data)
        final_pipeline = self.build_pipeline(data, selected_model)
        logger.info("Training final selected model on full training data.")
        final_pipeline.fit(x_train, y_train)
        final_pipeline.churn_model_name = selected_name
        final_pipeline.churn_threshold = selected_threshold
        final_pipeline.candidate_scores = self.candidate_scores
        return final_pipeline

    def train(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
        self.validate_training_data(data)
        x, y = self.split_features_and_target(data)
        y = self.encode_target(y)
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)
        trained_model = self.fit_final_model(data, x_train, y_train)
        return x_train, x_test, y_train, y_test, trained_model


@step
def train_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    trainer = TelcoChurnModelTrainer()
    trainer.validate_training_data(data)
    x, y = trainer.split_features_and_target(data)
    y = trainer.encode_target(y)
    x_train, x_test, y_train, y_test = trainer.split_train_test(x, y)
    trained_model = trainer.fit_final_model(data, x_train, y_train)
    return x_train, x_test, y_train, y_test, trained_model
