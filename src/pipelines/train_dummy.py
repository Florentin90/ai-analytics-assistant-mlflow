import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np

def run_experiment():
    mlflow.set_experiment("demo_experiment")

    with mlflow.start_run():
        # Dummy data
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = LinearRegression()
        model.fit(X, y)

        # Log parameter
        mlflow.log_param("model_type", "LinearRegression")

        # Log metric
        mlflow.log_metric("score", model.score(X, y))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Experiment logged successfully!")

if __name__ == "__main__":
    run_experiment()
