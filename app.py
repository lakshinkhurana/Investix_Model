from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import json

app = FastAPI()

# Load LSTM Model
model = tf.keras.models.load_model(r"c:\Users\Lakshin Khurana\vs-code-files\project\LSTM_model.keras", compile=False)

# Request Model for Prediction
class PredictionRequest(BaseModel):
    features: list[float]  # Expecting a list of float values

@app.get("/predict")
def predict(data: PredictionRequest):
    # Convert input list to NumPy array
    input_data = np.array(data.features).reshape(1, -1)

    # Get predictions from LSTM model
    prediction = model.predict(input_data)

    # Convert NumPy array to list
    prediction_list = prediction.tolist()

    # Convert to JSON format
    json_response = json.dumps({"prediction": prediction_list})

    return json_response


