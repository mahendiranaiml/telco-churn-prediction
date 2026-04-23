from zenml import pipeline

from src.evaluation import evaluate_model
from src.ingestion import ingest_data
from src.prediction import predict_model
from src.preprocessing import preprocess_data
from src.training import train_model


@pipeline
def training_pipeline(data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    raw_data = ingest_data(data_path)
    processed_data = preprocess_data(raw_data)
    x_train, x_test, y_train, y_test, trained_model = train_model(processed_data)
    x_train, x_test, y_train, y_test, y_pred, y_pred_proba = predict_model(
        x_train,
        x_test,
        y_train,
        y_test,
        trained_model,
    )
    return evaluate_model(
        x_train,
        x_test,
        y_train,
        y_test,
        y_pred,
        y_pred_proba,
        trained_model,
    )


if __name__ == "__main__":
    training_pipeline()
