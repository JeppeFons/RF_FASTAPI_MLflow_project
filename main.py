from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Indlæs modellen
model = joblib.load("rf_model.pkl")

# FastAPI-app
app = FastAPI(title="Credit Card Fraud Detection API")

@app.get("/")
def read_root():
    return {"message": "Velkommen til Credit Card Fraud Detection API. Gå til /docs for at se dokumentation."}


# Definér Pydantic-model for input
class Transaction(BaseModel):
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
    Amount_scaled: float
    Time_scaled: float

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Konverter input til DataFrame
        input_df = pd.DataFrame([transaction.dict()])
        
        print("Input shape:", input_df.shape)
        print("Input columns:", input_df.columns.tolist())

        # Lav prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        }

    except Exception as e:
        # Returnér fejlbesked
        return {"error": str(e)}
