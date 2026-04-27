from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler_new.pkl")

EXPECTED_FEATURES = scaler.n_features_in_


class TransactionInput(BaseModel):
    features: list[float]


def generate_explanation(fraud_probability, prediction):
    if prediction == 1:
        if fraud_probability >= 0.8:
            return (
                "This transaction has been classified as fraudulent because the model detected a very high fraud probability. "
                "The transaction pattern is significantly different from normal transaction behaviour and should be reviewed immediately."
            )
        else:
            return (
                "This transaction has been classified as potentially fraudulent. "
                "The model detected unusual transaction patterns, so further verification is recommended."
            )
    else:
        if fraud_probability < 0.3:
            return (
                "This transaction has been classified as legitimate because the fraud probability is low. "
                "The transaction pattern appears to be consistent with normal behaviour."
            )
        else:
            return (
                "This transaction is currently classified as legitimate, but the fraud probability is moderate. "
                "It may be useful to monitor this transaction if similar patterns continue."
            )


@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "scaler_loaded": True
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "Enhanced Random Forest",
        "expected_number_of_features": int(EXPECTED_FEATURES),
        "output": "Fraud probability, prediction label, risk band, recommended action, and explanation"
    }


@app.get("/sample-input")
def sample_input():
    return {
        "features": [0.0] * int(EXPECTED_FEATURES)
    }


@app.post("/predict")
def predict(transaction: TransactionInput):
    if len(transaction.features) != EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_FEATURES} features, but received {len(transaction.features)}"
        )

    data = np.array(transaction.features).reshape(1, -1)
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    fraud_probability = model.predict_proba(data_scaled)[0][1]

    if fraud_probability >= 0.8:
        risk_band = "High Risk"
        recommended_action = "Block transaction or send for immediate manual review"
    elif fraud_probability >= 0.5:
        risk_band = "Medium Risk"
        recommended_action = "Request additional verification"
    else:
        risk_band = "Low Risk"
        recommended_action = "Allow transaction"

    explanation = generate_explanation(fraud_probability, prediction)

    return {
        "prediction": int(prediction),
        "label": "Fraudulent" if prediction == 1 else "Legitimate",
        "fraud_probability": round(float(fraud_probability), 4),
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "explanation": explanation
    }


@app.get("/demo-fraud")
def demo_fraud():
    return {
        "prediction": 1,
        "label": "Fraudulent",
        "fraud_probability": 0.91,
        "risk_band": "High Risk",
        "recommended_action": "Block transaction or send for immediate manual review",
        "explanation": (
            "This demo transaction is classified as fraudulent because the risk score is very high. "
            "The transaction should be blocked or reviewed by a fraud analyst before approval."
        ),
        "note": "This endpoint is for demonstration purposes only. The /predict endpoint uses the real trained model."
    }