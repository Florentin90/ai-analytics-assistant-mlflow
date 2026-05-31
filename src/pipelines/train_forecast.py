import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .preprocess import preprocess
from ..data.load_data import load_energy_data


def train_forecast():
    df = load_energy_data()
    df = preprocess(df)

    X = df[["hour", "dayofweek", "lag_1", "lag_2", "lag_48"]]
    y = df["consumption"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="energy_forecast_rf"):
        model.fit(X, y)
        preds = model.predict(X)
        rmse = mean_squared_error(y, preds) ** 0.5

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)

        mlflow.sklearn.log_model(model, "model")

    print(f"Training complete. RMSE: {rmse:.2f}")
