import mlflow
import pandas as pd
from pathlib import Path

def load_model():
    # Find all MLflow model folders
    candidates = list(Path("mlruns").glob("**/artifacts/model"))
    if not candidates:
        raise RuntimeError("No MLflow model found. Train a model first.")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return mlflow.sklearn.load_model(str(latest))

def prepare_input(timestamp: str, lag_1: float, lag_2: float, lag_48: float):
    ts = pd.to_datetime(timestamp)

    df = pd.DataFrame([{
        "timestamp": ts,
        "hour": ts.hour,
        "dayofweek": ts.dayofweek,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_48": lag_48
    }])

    return df

def predict_consumption(timestamp: str, lag_1: float, lag_2: float, lag_48: float):
    model = load_model()
    df = prepare_input(timestamp, lag_1, lag_2, lag_48)
    pred = model.predict(df)[0]
    return float(pred)