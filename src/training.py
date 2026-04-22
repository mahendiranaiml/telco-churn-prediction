import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from zenml import step

from src.preprocessing import TARGET_COLUMN, TelcoDataPreprocessor


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": DEFAULT_RANDOM_STATE,
}


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
        random_state: int = DEFAULT_RANDOM_STATE,
        model_params: dict | None = None,
    ) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model_params = model_params or DEFAULT_MODEL_PARAMS.copy()
        self.label_encoder = LabelEncoder()

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

    def build_model(self) -> RandomForestClassifier:
        params = self.model_params.copy()
        params.setdefault("random_state", self.random_state)
        logger.info("Building RandomForestClassifier with params: %s", params)
        return RandomForestClassifier(**params)

    def build_training_pipeline(self, data: pd.DataFrame) -> Pipeline:
        preprocessor = TelcoDataPreprocessor().build_column_transformer(data)
        model = self.build_model()
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def fit_model(self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        logger.info("Training model...")
        model.fit(x_train, y_train)
        logger.info("Training completed.")
        return model

    def train(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
        self.validate_training_data(data)
        x, y = self.split_features_and_target(data)
        y = self.encode_target(y)
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)
        model = self.build_training_pipeline(data)
        trained_model = self.fit_model(model, x_train, y_train)
        return x_train, x_test, y_train, y_test, trained_model


@step
def train_model(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    trainer = TelcoChurnModelTrainer()
    trainer.validate_training_data(data)
    x, y = trainer.split_features_and_target(data)
    y = trainer.encode_target(y)
    x_train, x_test, y_train, y_test = trainer.split_train_test(x, y)
    model = trainer.build_training_pipeline(data)
    trained_model = trainer.fit_model(model, x_train, y_train)
    return x_train, x_test, y_train, y_test, trained_model
