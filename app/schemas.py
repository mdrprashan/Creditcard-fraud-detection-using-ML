from typing import List

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    Time: float = Field(..., description="Seconds elapsed between this transaction and the first transaction.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in the source currency.")

    def to_feature_dict(self) -> dict[str, float]:
        return self.dict()


class PredictionResult(BaseModel):
    fraud_probability: float
    predicted_label: int
    risk_band: str
    threshold_used: float


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    threshold: float
    feature_count: int
    feature_names: List[str]
    metrics: dict
