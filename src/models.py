# src/models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_model(name, params=None):
    params = {key: value for key, value in (params or {}).items() if key != "type"}
    if name == "LinearRegression":
        return LinearRegression(**params)
    if name == "RandomForest":
        return RandomForestRegressor(**params)
    if name == "XGBoost":
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model: {name}")