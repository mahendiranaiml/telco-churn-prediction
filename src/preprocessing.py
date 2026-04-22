import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import step


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TOTAL_CHARGES_COLUMN = "TotalCharges"
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


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

    def simplify_service_values(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned = data.copy()
        cleaned = cleaned.replace(
            {
                "No internet service": "No",
                "No phone service": "No",
            }
        )
        logger.info("Simplified service values for no internet/phone service.")
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
        featured["avg_charge"] = featured[TOTAL_CHARGES_COLUMN] / (featured["tenure"] + 1)
        featured["is_month_to_month"] = (featured["Contract"] == "Month-to-month").astype(int)
        featured["tenure_group"] = pd.cut(
            featured["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        ).astype(str)
        featured["num_services"] = featured[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        logger.info("Added engineered features: tenure_group, avg_charge, num_services.")
        return featured

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input_data(data)
        data = self.drop_customer_id(data)
        data = self.clean_total_charges(data)
        data = self.simplify_service_values(data)
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

        for column in SERVICE_COLUMNS + ["MultipleLines"]:
            if transformed.get(column) in {"No internet service", "No phone service"}:
                transformed[column] = "No"

        transformed[TOTAL_CHARGES_COLUMN] = total_charges
        transformed["avg_monthly_spend"] = total_charges / tenure
        transformed["avg_charge"] = total_charges / (tenure + 1)
        transformed["is_month_to_month"] = int(transformed["Contract"] == "Month-to-month")
        transformed["tenure_group"] = self.get_tenure_group(tenure)
        transformed["num_services"] = sum(1 for column in SERVICE_COLUMNS if transformed.get(column) == "Yes")
        return transformed

    def get_tenure_group(self, tenure: int) -> str:
        if tenure <= 12:
            return "0-1yr"
        if tenure <= 24:
            return "1-2yr"
        if tenure <= 48:
            return "2-4yr"
        return "4-6yr"


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    preprocessor = TelcoDataPreprocessor()
    preprocessor.validate_input_data(data)
    data = preprocessor.drop_customer_id(data)
    data = preprocessor.clean_total_charges(data)
    data = preprocessor.simplify_service_values(data)
    data = preprocessor.remove_duplicates(data)
    data = preprocessor.add_features(data)
    logger.info("Preprocessing step completed with %s rows and %s columns.", data.shape[0], data.shape[1])
    return data
