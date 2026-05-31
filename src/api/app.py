from fastapi import FastAPI
from src.pipelines.train_dummy import run_experiment

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run-experiment")
def run_mlflow_experiment():
    run_experiment()
    return {"message": "Experiment triggered successfully"}
