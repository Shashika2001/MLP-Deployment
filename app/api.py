from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train models with "
        "`python -m src.train` from the repository root."
    )

app = FastAPI(title="House Price Prediction API")
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict(features: dict):
    values = np.array([list(features.values())]).reshape(1, -1)
    prediction = model.predict(values)[0]
    return {"predicted_price": float(prediction * 100000)}