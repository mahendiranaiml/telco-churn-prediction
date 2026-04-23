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


def get_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.preprocessing")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger()


class DataPreprocessing(ABC):
    @abstractmethod
    def validate_input_data(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def prepare_customer_data(self, customer_data: dict) -> dict:
        pass


class TelcoDataPreprocessing(DataPreprocessing):
    def validate_input_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Cannot preprocess an empty dataset")
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"Target column is missing: {TARGET_COLUMN}")
        if TOTAL_CHARGES_COLUMN not in data.columns:
            raise ValueError(f"Required column is missing: {TOTAL_CHARGES_COLUMN}")

    def clean_total_charges(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data[TOTAL_CHARGES_COLUMN] = pd.to_numeric(data[TOTAL_CHARGES_COLUMN], errors="coerce")
        missing_count = int(data[TOTAL_CHARGES_COLUMN].isna().sum())

        if missing_count:
            median_value = float(data[TOTAL_CHARGES_COLUMN].median())
            logger.info("Filling %s missing TotalCharges values with median value %.2f.", missing_count, median_value)
            data[TOTAL_CHARGES_COLUMN] = data[TOTAL_CHARGES_COLUMN].fillna(median_value)

        return data

    def simplify_service_values(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data = data.replace({"No internet service": "No", "No phone service": "No"})
        logger.info("Simplified service values for no internet/phone service.")
        return data

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        safe_tenure = data["tenure"].replace(0, 1)
        data["avg_monthly_spend"] = data[TOTAL_CHARGES_COLUMN] / safe_tenure
        data["avg_charge"] = data[TOTAL_CHARGES_COLUMN] / (data["tenure"] + 1)
        data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(int)
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        ).astype(str)
        data["num_services"] = data[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        logger.info("Added engineered features: tenure_group, avg_charge, num_services.")
        return data

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        self.validate_input_data(data)
        data = data.copy()

        if ID_COLUMN in data.columns:
            logger.info("Dropping identifier column: %s", ID_COLUMN)
            data = data.drop(columns=[ID_COLUMN])

        data = self.clean_total_charges(data)
        data = self.simplify_service_values(data)

        duplicate_count = int(data.duplicated().sum())
        if duplicate_count:
            logger.info("Removing %s duplicate rows.", duplicate_count)
            data = data.drop_duplicates()

        data = self.add_features(data)
        logger.info("Preprocessing completed with %s rows and %s columns.", data.shape[0], data.shape[1])
        return data

    def prepare_customer_data(self, customer_data: dict) -> dict:
        customer_data = customer_data.copy()
        safe_tenure = int(customer_data.get("tenure", 0)) or 1
        total_charges = float(customer_data[TOTAL_CHARGES_COLUMN])

        for column in SERVICE_COLUMNS + ["MultipleLines"]:
            if customer_data.get(column) in {"No internet service", "No phone service"}:
                customer_data[column] = "No"

        customer_data[TOTAL_CHARGES_COLUMN] = total_charges
        customer_data["avg_monthly_spend"] = total_charges / safe_tenure
        customer_data["avg_charge"] = total_charges / (safe_tenure + 1)
        customer_data["is_month_to_month"] = int(customer_data["Contract"] == "Month-to-month")
        customer_data["tenure_group"] = pd.cut(
            pd.Series([safe_tenure]),
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        ).astype(str).iloc[0]
        customer_data["num_services"] = sum(1 for column in SERVICE_COLUMNS if customer_data.get(column) == "Yes")
        return customer_data


def build_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
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


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    preprocessing = TelcoDataPreprocessing()
    processed_data = preprocessing.preprocess(data)
    logger.info("Preprocessing step completed with %s rows and %s columns.", processed_data.shape[0], processed_data.shape[1])
    return processed_data
