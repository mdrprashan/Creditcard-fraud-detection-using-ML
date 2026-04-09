from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model_service import FraudDetectionService
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    TransactionFeatures,
)


service = FraudDetectionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        service.load()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Production-style inference API for predicting potentially fraudulent credit card transactions.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.is_loaded())


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")

    return ModelInfoResponse(
        model_name=service.model_name,
        threshold=service.threshold,
        feature_count=len(service.feature_names),
        feature_names=service.feature_names,
        metrics=service.metrics,
    )


@app.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionFeatures) -> PredictionResult:
    try:
        return service.predict_one(transaction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        predictions = service.predict_many(request.transactions)
        return BatchPredictionResponse(predictions=predictions)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
