import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "weave_energy.parquet"

def load_energy_data():
    df = pd.read_parquet(DATA_PATH)

    df = df.rename(columns={
        "data_collection_log_timestamp": "timestamp",
        "total_consumption_active_import": "consumption"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df
