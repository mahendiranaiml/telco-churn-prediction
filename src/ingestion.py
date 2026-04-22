import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from zenml import step

DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

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


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("telco_churn.ingestion")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logger()


class DatasetIngestor(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load a dataset and returns pandas DataFrame."""


class CSVDatasetIngestor(DatasetIngestor):
    def __init__(self, file_path: str | Path = DEFAULT_DATA_PATH) -> None:
        self.file_path = self._resolve_path(Path(file_path))

    def _resolve_path(self, file_path: Path) -> Path:
        if file_path.exists():
            return file_path

        raw_dir = Path("data/raw")
        csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
        if len(csv_files) == 1:
            logger.warning("Configured file %s not found. Using %s instead.", file_path, csv_files[0])
            return csv_files[0]

        raise FileNotFoundError(f"CSV file not found: {file_path}")

    def validate_file(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {self.file_path.suffix}")

        if self.file_path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {self.file_path}")

    def read_csv(self) -> pd.DataFrame:
        logger.info("Loading dataset from %s", self.file_path)

        try:
            return pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV file has no readable data: {self.file_path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV file is not correctly formatted: {self.file_path}") from exc

    def validate_dataframe(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Loaded dataset is empty")

        if data.columns.empty:
            raise ValueError("Loaded dataset has no columns")

        missing_columns = REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        duplicate_rows = data.duplicated().sum()
        if duplicate_rows:
            logger.warning("Dataset contains %s duplicate rows.", duplicate_rows)

        null_counts = data.isna().sum()
        columns_with_nulls = null_counts[null_counts > 0].to_dict()
        if columns_with_nulls:
            logger.warning("Dataset contains missing values: %s", columns_with_nulls)

    def load_data(self) -> pd.DataFrame:
        self.validate_file()
        data = self.read_csv()
        self.validate_dataframe(data)
        logger.info("Dataset loaded successfully with %s rows and %s columns.", data.shape[0], data.shape[1])
        return data


@step
def ingest_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    ingestor = CSVDatasetIngestor(path)
    ingestor.validate_file()
    data = ingestor.read_csv()
    ingestor.validate_dataframe(data)
    logger.info("Ingestion step completed with %s rows and %s columns.", data.shape[0], data.shape[1])
    return data
