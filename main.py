from fastapi import FastAPI
from pydantic import BaseModel
from src.pipelines.predict_forecast import predict_consumption

app = FastAPI()

class PredictRequest(BaseModel):
    timestamp: str
    lag_1: float
    lag_2: float
    lag_48: float

@app.post("/predict")
def predict(req: PredictRequest):
    prediction = predict_consumption(
        req.timestamp,
        req.lag_1,
        req.lag_2,
        req.lag_48
    )
    return {"prediction": prediction}
