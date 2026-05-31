import pandas as pd

def preprocess(df: pd.DataFrame):
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Remove impossible or corrupt values
    df = df[df["consumption"] > 0].copy()

    # Time-based features
    df.loc[:, "hour"] = df["timestamp"].dt.hour
    df.loc[:, "dayofweek"] = df["timestamp"].dt.dayofweek

    # Lag features for forecasting
    df.loc[:, "lag_1"] = df["consumption"].shift(1)
    df.loc[:, "lag_2"] = df["consumption"].shift(2)
    df.loc[:, "lag_48"] = df["consumption"].shift(48)

    # Drop rows with NaN created by lagging
    df = df.dropna()

    return df
