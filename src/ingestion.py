import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from zenml import step


DEFAULT_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def get_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.ingestion")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


logger = get_logger()


class DataIngestion(ABC):
    @abstractmethod
    def validate_file(self) -> None:
        pass

    @abstractmethod
    def read_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass


class CSVDataIngestion(DataIngestion):
    def __init__(self, file_path: str = DEFAULT_DATA_PATH):
        self.file_path = self.resolve_data_path(file_path)

    def resolve_data_path(self, file_path: str) -> Path:
        selected_path = Path(file_path)
        if selected_path.exists():
            return selected_path

        raw_directory = Path("data/raw")
        if raw_directory.exists():
            csv_files = list(raw_directory.glob("*.csv"))
            csv_files.sort()
            if len(csv_files) == 1:
                fallback_path = csv_files[0]
                logger.warning("Configured file %s not found. Using %s instead.", selected_path, fallback_path)
                return fallback_path

        raise FileNotFoundError(f"CSV file not found: {selected_path}")

    def validate_file(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {self.file_path.suffix}")
        if self.file_path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {self.file_path}")

    def read_data(self) -> pd.DataFrame:
        logger.info("Loading dataset from %s", self.file_path)
        try:
            return pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV file has no readable data: {self.file_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV file is not correctly formatted: {self.file_path}") from exc

    def validate_data(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Loaded dataset is empty")

        missing_columns = REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        duplicate_rows = int(data.duplicated().sum())
        if duplicate_rows:
            logger.warning("Dataset contains %s duplicate rows.", duplicate_rows)

        missing_values = data.isna().sum()
        missing_summary = missing_values[missing_values > 0].to_dict()
        if missing_summary:
            logger.warning("Dataset contains missing values: %s", missing_summary)

    def load_data(self) -> pd.DataFrame:
        self.validate_file()
        data = self.read_data()
        self.validate_data(data)
        logger.info("Dataset loaded successfully with %s rows and %s columns.", data.shape[0], data.shape[1])
        return data


@step
def ingest_data(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    ingestion = CSVDataIngestion(file_path)
    data = ingestion.load_data()
    logger.info("Ingestion step completed with %s rows and %s columns.", data.shape[0], data.shape[1])
    return data
