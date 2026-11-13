from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("California House Price Predictor")
st.markdown(
    """
    This app predicts **median house prices** using a model trained and tracked
    with **MLflow**. The model shown below is the *best-performing* one based on
    R² score across multiple experiments.
    """
)

if not MODEL_PATH.is_file():
    st.error(
        "Model not found. Train models first with "
        "`python -m src.train` from the repository root."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("Best model loaded successfully!")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    client = MlflowClient()
    experiment = client.get_experiment_by_name("house_price_prediction")
    if experiment:
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["metrics.R2 DESC"],
            max_results=5,
        )
        st.subheader("Top Experiment Runs (from MLflow)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": run.data.params.get("model_name", ""),
                        "R2": run.data.metrics.get("R2", 0),
                        "RMSE": run.data.metrics.get("RMSE", 0),
                    }
                    for run in runs
                ]
            )
        )
except Exception as error:
    st.warning(f"Could not fetch MLflow experiments: {error}")

st.sidebar.header("Input Features")
median_income = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 5.0)
house_age = st.sidebar.slider("House Age (years)", 1, 50, 20)
average_rooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
average_bedrooms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.sidebar.slider("Population", 100, 5000, 800)
average_occupants = st.sidebar.slider("Average Occupants", 1.0, 10.0, 3.0)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 36.5)
longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -119.0)
input_data = np.array(
    [[
        median_income,
        house_age,
        average_rooms,
        average_bedrooms,
        population,
        average_occupants,
        latitude,
        longitude,
    ]]
)

if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.subheader("Predicted Median House Value")
    st.metric(label="Estimated Price", value=f"${prediction * 100000:,.2f}")