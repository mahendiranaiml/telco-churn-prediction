import logging
from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Annotated
from zenml import step

from src.preprocessing import TARGET_COLUMN, build_preprocessor


DEFAULT_CONFIG_PATH = "config.yaml"
def get_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.training")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger()


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


class ModelTraining(ABC):
    @abstractmethod
    def validate_training_data(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame):
        pass


class TelcoModelTraining(ModelTraining):
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]

    def validate_training_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Training data is empty")
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column not found: {TARGET_COLUMN}")
        if data[TARGET_COLUMN].nunique() < 2:
            raise ValueError("Target column must contain at least two classes")

    def split_features_and_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]
        logger.info("Created feature matrix with %s columns.", x.shape[1])
        return x, y

    def encode_target(self, y: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
        logger.info("Encoded target classes: %s", list(encoder.classes_))
        return encoded, encoder

    def split_train_test(self, x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.data_config["test_size"],
            random_state=self.data_config["random_state"],
            stratify=y,
        )
        logger.info("Train shape: %s, test shape: %s", x_train.shape, x_test.shape)
        return x_train, x_test, y_train, y_test

    def get_candidate_models(self) -> dict:
        params = self.model_config["params"]

        logistic_params = params["logistic_regression"].copy()
        random_forest_params = params["random_forest"].copy()
        gradient_boosting_params = params["gradient_boosting"].copy()

        return {
            "logistic_regression": LogisticRegression(**logistic_params),
            "random_forest": RandomForestClassifier(**random_forest_params),
            "gradient_boosting": GradientBoostingClassifier(**gradient_boosting_params),
        }

    def get_selected_model(self) -> Tuple[str, object]:
        model_name = self.model_config["type"]
        models = self.get_candidate_models()

        if model_name not in models:
            raise ValueError(f"Unsupported model type: {model_name}")

        return model_name, models[model_name]

    def build_pipeline(self, data: pd.DataFrame, model: object) -> Pipeline:
        return Pipeline([("preprocessor", build_preprocessor(data)), ("model", model)])

    def train(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
        self.validate_training_data(data)
        x, y = self.split_features_and_target(data)
        y, encoder = self.encode_target(y)
        x_train, x_test, y_train, y_test = self.split_train_test(x, y)

        model_name, model = self.get_selected_model()
        threshold = float(self.training_config["threshold"])
        final_pipeline = self.build_pipeline(data, model)
        logger.info("Training selected model: %s", model_name)
        final_pipeline.fit(x_train, y_train)
        final_pipeline.churn_model_name = model_name
        final_pipeline.churn_threshold = threshold
        final_pipeline.label_encoder = encoder
        return x_train, x_test, y_train, y_test, final_pipeline


@step
def train_model(
    data: pd.DataFrame,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[Pipeline, "trained_model"],
]:
    training = TelcoModelTraining(config_path)
    x_train, x_test, y_train, y_test, trained_model = training.train(data)
    return x_train, x_test, y_train, y_test, trained_model
