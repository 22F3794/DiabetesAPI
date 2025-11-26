# ====================================================
# main.py - Diabetes Prediction API (Production)
# ====================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib

# -----------------------------
# Load Trained Model & Scaler
# -----------------------------
# Make sure these files are in the same folder as main.py
scaler = joblib.load("scaler.pkl")
best_model = joblib.load("diabetes_model.pkl")

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes based on patient data using trained ML model",
    version="1.0"
)

# -----------------------------
# Enable CORS (so mobile app can call API)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Define Input Data Model
# -----------------------------
class InputData(BaseModel):
    Age: float
    Sex: float
    BMI: float
    Glucose: float
    BloodPressure: float
    Insulin: float
    Increased_Thirst: float
    Increased_Hunger: float
    Fatigue_Tiredness: float
    Blurred_Vision: float
    Unexplained_Weight_Loss: float

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    # Convert input to numpy array
    x = np.array([[
        data.Age,
        data.Sex,
        data.BMI,
        data.Glucose,
        data.BloodPressure,
        data.Insulin,
        data.Increased_Thirst,
        data.Increased_Hunger,
        data.Fatigue_Tiredness,
        data.Blurred_Vision,
        data.Unexplained_Weight_Loss
    ]])

    # Scale input using trained scaler
    x_scaled = scaler.transform(x)

    # Make prediction using trained model
    pred = best_model.predict(x_scaled)[0]

    # Map prediction to human-readable output
    result = "Diabetes" if pred == 1 else "No Diabetes"
    return {"prediction": result}

# -----------------------------
# Run API
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
