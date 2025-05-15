from fastapi import FastAPI
import pandas as pd
import numpy as np
from pydantic import BaseModel
import uvicorn
import os
import joblib

# Define FastAPI app
app = FastAPI()

# Load the model
model = joblib.load('models/Randomforest_model.pkl')

print("model loaded successfully")

# Define the FastAPI app
app = FastAPI(title="Churn Prediction API")

# Input schema using Pydantic
class UserInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Churn Prediction API is running."}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: UserInput):
    input_df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return {"prediction": "Churn" if prediction else "No Churn", "churn_probability": round(proba, 3)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)