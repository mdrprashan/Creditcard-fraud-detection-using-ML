# Credit Card Fraud Detection API

This project packages your fraud detection work into a more realistic machine learning application for a Masters project. It includes:

- a training pipeline that builds and saves a reusable fraud detection model
- a FastAPI service for single and batch transaction scoring
- model metadata and health endpoints for operational visibility

## Why this is closer to a real-world system

Instead of only showing notebook experiments, this repository now supports a common production workflow:

1. Train a model from historical transactions
2. Save the model as a versioned artifact
3. Expose an API that other systems can call to score new transactions
4. Return a fraud probability, prediction label, and risk band

This is the kind of structure you can describe in your dissertation, demo in a viva, or extend into a cloud deployment later.

## Project Structure

```text
app/
  main.py            # FastAPI application
  model_service.py   # Loads model artifact and performs inference
  schemas.py         # Request/response models
data/
  raw/creditcard.csv # Source dataset
models/              # Saved trained model artifacts
src/
  train_model.py     # Training script that saves the model
main.py              # ASGI entrypoint
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py
```

This reads `data/raw/creditcard.csv` and writes the trained artifact to `models/fraud_detection_model.joblib`.

## Run the API

```bash
uvicorn app.main:app --reload
```

Swagger docs will be available at:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### `GET /health`

Checks whether the API is running and whether a trained model has been loaded.

### `GET /model/info`

Returns:

- model name
- threshold used for fraud classification
- feature names
- evaluation metrics captured at training time

### `POST /predict`

Scores a single transaction.

Example request body:

```json
{
  "Time": 0,
  "V1": -1.3598,
  "V2": -0.0727,
  "V3": 2.5363,
  "V4": 1.3781,
  "V5": -0.3383,
  "V6": 0.4623,
  "V7": 0.2395,
  "V8": 0.0986,
  "V9": 0.3637,
  "V10": 0.0907,
  "V11": -0.5515,
  "V12": -0.6178,
  "V13": -0.9913,
  "V14": -0.3111,
  "V15": 1.4681,
  "V16": -0.4704,
  "V17": 0.2079,
  "V18": 0.0257,
  "V19": 0.4039,
  "V20": 0.2514,
  "V21": -0.0183,
  "V22": 0.2778,
  "V23": -0.1104,
  "V24": 0.0669,
  "V25": 0.1285,
  "V26": -0.1891,
  "V27": 0.1335,
  "V28": -0.0210,
  "Amount": 149.62
}
```

Example response:

```json
{
  "fraud_probability": 0.017421,
  "predicted_label": 0,
  "risk_band": "low",
  "threshold_used": 0.5
}
```

### `POST /predict/batch`

Scores multiple transactions in one request.

## Good Masters project extensions

If you want to make this feel even more industry-grade, the strongest next additions would be:

- threshold tuning based on business cost of false positives vs false negatives
- model versioning and experiment tracking with MLflow
- feature store style preprocessing shared by training and inference
- database logging of scored transactions and outcomes
- authentication and rate limiting for the API
- Docker deployment
- drift monitoring and retraining workflow

## Important project note

This dataset uses anonymized PCA-based features (`V1` to `V28`), which is excellent for learning but not identical to live bank data pipelines. In your report, it is worth saying that this API demonstrates the architecture of a fraud detection service, while a real bank would add streaming ingestion, stronger governance, richer merchant/device features, and human review workflows.
