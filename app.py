from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipelines.inference_pipeline import inference_pipeline
from fastapi.responses import FileResponse


class ChurnPredictionRequest(BaseModel):
    gender: str
    SeniorCitizen: int = Field(alias="senior_citizen", ge=0, le=1)
    Partner: str = Field(alias="partner")
    Dependents: str = Field(alias="dependents")
    tenure: int = Field(ge=0)
    PhoneService: str = Field(alias="phone_service")
    MultipleLines: str = Field(alias="multiple_lines")
    InternetService: str = Field(alias="internet_service")
    OnlineSecurity: str = Field(alias="online_security")
    OnlineBackup: str = Field(alias="online_backup")
    DeviceProtection: str = Field(alias="device_protection")
    TechSupport: str = Field(alias="tech_support")
    StreamingTV: str = Field(alias="streaming_tv")
    StreamingMovies: str = Field(alias="streaming_movies")
    Contract: str = Field(alias="contract")
    PaperlessBilling: str = Field(alias="paperless_billing")
    PaymentMethod: str = Field(alias="payment_method")
    MonthlyCharges: float = Field(alias="monthly_charges", ge=0)
    TotalCharges: float = Field(alias="total_charges", ge=0)

    model_config = {"populate_by_name": True}


app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict_churn(request: ChurnPredictionRequest) -> dict:
    try:
        payload = request.model_dump(by_alias=False)
        return inference_pipeline(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
