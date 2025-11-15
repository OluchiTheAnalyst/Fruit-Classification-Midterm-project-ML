from fastapi import FastAPI
import pickle
import pandas as pd

# Initialize app
app = FastAPI(title="Fruit Classification API", version="1.0")

# Load saved pipeline
with open("model.bin", "rb") as f_in:
    encoder, dv, model = pickle.load(f_in)

# Root endpoint (health check)
@app.get("/")
def home():
    return {"message": "Fruit Classification API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict(fruit: dict):
   
    # Convert input dictionary to DataFrame
    X = dv.transform([fruit])

    # Predict numeric class
    y_pred = model.predict(X)[0]

    # Decode numeric label back to fruit name
    fruit_name = encoder.inverse_transform([y_pred])[0]

    return {"predicted_fruit": fruit_name}

