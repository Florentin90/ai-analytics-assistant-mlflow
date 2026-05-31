import pandas as pd

print("Downloading filtered Weave dataset...")

df = pd.read_parquet(
    "s3://weave.energy/smart-meter",
    filters=[("data_collection_log_timestamp", "==", pd.Timestamp("2024-07-14T20:00:00Z"))]
)

df.to_parquet("weave_energy.parquet")

print("Saved weave_energy.parquet")
