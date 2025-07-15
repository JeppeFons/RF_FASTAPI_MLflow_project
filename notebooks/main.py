from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import numpy as np

# === Indlæs den færdigtrænede model fra MLflow artifacts ===
run_id = "e835ced6b4dd41e98d2ddb6b18e3220c" # Udskift RUN_ID med den faktiske run-id for din final model
model_uri = f"runs:/{run_id}/model"  
model = mlflow.sklearn.load_model(model_uri)

# === FastAPI instans ===
app = FastAPI(title="Credit Card Fraud Detection API")

# === Pydantic model til input ===
class Transaction(BaseModel):
    # Tilføj alle features her som float. Eksempel:
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "API kører – send POST-anmodninger til /predict"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Konverter til DataFrame
    input_df = pd.DataFrame([transaction.dict()])

    # Forudsig
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_fraud": round(float(probability), 4)
    }