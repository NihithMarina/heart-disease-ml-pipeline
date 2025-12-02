from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

# Input Schema
class InputSchema(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load saved model
model = joblib.load("model_pipeline.pkl")

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputSchema):
    data = pd.DataFrame([input_data.dict()])
    y_proba = model.predict_proba(data)[0][1]
    y_pred = model.predict(data)[0]

    return {
        "prediction": int(y_pred),
        "probability": float(y_proba)
    }
