from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from app.schemas import PredictionResult, TransactionFeatures


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.joblib"


class FraudDetectionService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = None
        self.threshold = 0.5
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self.model_name = "unloaded"

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run `python src/train_model.py` first."
            )

        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact["metrics"]
        self.model_name = artifact["model_name"]

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_one(self, transaction: TransactionFeatures) -> PredictionResult:
        return self.predict_many([transaction])[0]

    def predict_many(self, transactions: Iterable[TransactionFeatures]) -> list[PredictionResult]:
        if not self.is_loaded():
            self.load()

        rows = [transaction.to_feature_dict() for transaction in transactions]
        frame = pd.DataFrame(rows)
        frame = frame[self.feature_names]

        probabilities = self.model.predict_proba(frame)[:, 1]
        predictions = []

        for probability in probabilities:
            predicted_label = int(probability >= self.threshold)
            predictions.append(
                PredictionResult(
                    fraud_probability=round(float(probability), 6),
                    predicted_label=predicted_label,
                    risk_band=self._risk_band(float(probability)),
                    threshold_used=self.threshold,
                )
            )

        return predictions

    @staticmethod
    def _risk_band(probability: float) -> str:
        if probability >= 0.9:
            return "critical"
        if probability >= 0.7:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"
