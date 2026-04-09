from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_detection_model.joblib"
TARGET_COLUMN = "Class"
DEFAULT_THRESHOLD = 0.5


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def evaluate(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_test, probabilities)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "test_size": int(len(y_test)),
        "fraud_cases_in_test": int(y_test.sum()),
    }


def train() -> Path:
    dataset = load_dataset(DATA_PATH)
    x = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)

    artifact = {
        "model": model,
        "feature_names": list(x.columns),
        "threshold": DEFAULT_THRESHOLD,
        "metrics": metrics,
        "model_name": "RandomForestClassifier",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"Saved model artifact to {MODEL_PATH}")
    print(f"Metrics: {metrics}")
    return MODEL_PATH


if __name__ == "__main__":
    train()
