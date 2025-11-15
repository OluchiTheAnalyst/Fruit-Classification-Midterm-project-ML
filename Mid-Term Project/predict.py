import pickle
import pandas as pd

# Load your pipeline
with open("model.bin", "rb") as f_in:
    encoder, dv, model = pickle.load(f_in)

# Example fruit to classify
fruit = {
    "size (cm)": 7.5,
    "shape": "round",
    "weight (g)": 250.5,
    "avg_price": 80.4,
    "color": "green",
    "taste": "sweet"
}

# Convert dictionary to vector
X = dv.transform([fruit])

# Predict numeric label
y_pred = model.predict(X)[0]

# Decode numeric label back to fruit name
fruit_name = encoder.inverse_transform([y_pred])[0]

print(f"Predicted Fruit: {fruit_name}")