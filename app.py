from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import requests
import json

app = FastAPI()

# Load LSTM Model
model = tf.keras.models.load_model(r"c:\Users\Lakshin Khurana\vs-code-files\project\LSTM_model.keras", compile=False)

# Request Model for Prediction
class PredictionRequest(BaseModel):
    features: list[float]  # Expecting a list of float values

@app.get("/predict")
def predict(features: str):
    # Convert query string into a list of floats
    try:
        feature_list = [float(x) for x in features.split(",")]
    except ValueError:
        return {"error": "Invalid input format"}

    # Dummy prediction (replace with your model logic)
    prediction = sum(feature_list)  # Example: Just summing up the numbers

    return {"prediction": prediction}


@app.get("/")
async def home():
    return {"message": "Welcome to my API"}


url = "http://127.0.0.1:8000/predict"
params = {"features": "0.5,1.2,3.4"}  # Convert list to comma-separated string

response = requests.get(url, params=params)
print(response.json())

