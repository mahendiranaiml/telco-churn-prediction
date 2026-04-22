import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import step


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TOTAL_CHARGES_COLUMN = "TotalCharges"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.preprocessing")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()


class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data and return model-ready data."""


class TelcoDataPreprocessor(DataPreprocessor):
    def validate_input_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Cannot preprocess an empty dataset")

        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column is missing: {TARGET_COLUMN}")

        if TOTAL_CHARGES_COLUMN not in data.columns:
            raise ValueError(f"Required column is missing: {TOTAL_CHARGES_COLUMN}")

    def drop_customer_id(self, data: pd.DataFrame) -> pd.DataFrame:
        if ID_COLUMN in data.columns:
            logger.info("Dropping identifier column: %s", ID_COLUMN)
            return data.drop(columns=[ID_COLUMN])
        return data

    def clean_total_charges(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.copy()
        cleaned[TOTAL_CHARGES_COLUMN] = pd.to_numeric(cleaned[TOTAL_CHARGES_COLUMN], errors="coerce")

        missing_count = cleaned[TOTAL_CHARGES_COLUMN].isna().sum()
        if missing_count:
            median_value = cleaned[TOTAL_CHARGES_COLUMN].median()
            logger.info(
                "Filling %s missing TotalCharges values with median value %.2f.",
                missing_count,
                median_value,
            )
            cleaned[TOTAL_CHARGES_COLUMN] = cleaned[TOTAL_CHARGES_COLUMN].fillna(median_value)

        return cleaned

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        duplicate_count = data.duplicated().sum()
        if duplicate_count:
            logger.info("Removing %s duplicate rows.", duplicate_count)
            return data.drop_duplicates()
        return data

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        featured = data.copy()
        tenure = featured["tenure"].replace(0, 1)
        featured["avg_monthly_spend"] = featured[TOTAL_CHARGES_COLUMN] / tenure
        featured["is_month_to_month"] = (featured["Contract"] == "Month-to-month").astype(int)
        logger.info("Added engineered features: avg_monthly_spend, is_month_to_month")
        return featured

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input_data(data)
        data = self.drop_customer_id(data)
        data = self.clean_total_charges(data)
        data = self.remove_duplicates(data)
        data = self.add_features(data)
        logger.info("Preprocessing completed with %s rows and %s columns.", data.shape[0], data.shape[1])
        return data

    def build_column_transformer(self, data: pd.DataFrame) -> ColumnTransformer:
        features = data.drop(columns=[TARGET_COLUMN], errors="ignore")
        categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_features = features.select_dtypes(exclude=["object", "category"]).columns.tolist()

        logger.info("Categorical features: %s", categorical_features)
        logger.info("Numerical features: %s", numeric_features)

        return ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("numeric", StandardScaler(), numeric_features),
            ]
        )

    def preprocess_payload(self, payload: dict) -> dict:
        transformed = payload.copy()
        total_charges = float(transformed[TOTAL_CHARGES_COLUMN])
        tenure = int(transformed.get("tenure", 0)) or 1
        transformed[TOTAL_CHARGES_COLUMN] = total_charges
        transformed["avg_monthly_spend"] = total_charges / tenure
        transformed["is_month_to_month"] = int(transformed["Contract"] == "Month-to-month")
        return transformed


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    preprocessor = TelcoDataPreprocessor()
    preprocessor.validate_input_data(data)
    data = preprocessor.drop_customer_id(data)
    data = preprocessor.clean_total_charges(data)
    data = preprocessor.remove_duplicates(data)
    data = preprocessor.add_features(data)
    logger.info("Preprocessing step completed with %s rows and %s columns.", data.shape[0], data.shape[1])
    return data
