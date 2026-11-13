# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
from pathlib import Path
import yaml

from .data_loader import load_data
from .models import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
SKOPS_TRUSTED_TYPES = {
    "RandomForest": ["sklearn.tree._tree.Tree"],
    "XGBoost": ["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"],
}


def train(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "params.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    X_train, X_test, y_train, y_test = load_data(config_path)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config["experiment_name"])
    best_model, best_r2, best_name = None, float("-inf"), None
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    for name, params in config["models"].items():
        safe_params = dict(params or {})
        safe_params.pop("type", None)
        with mlflow.start_run(run_name=name):
            model = get_model(name, params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mlflow.log_param("model_name", name)
            mlflow.log_params(safe_params)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.sklearn.log_model(
                model,
                name=name,
                skops_trusted_types=SKOPS_TRUSTED_TYPES.get(name),
            )
            print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")

        if r2 > best_r2:
            best_model = model
            best_r2 = r2
            best_name = name
            best_model_path = models_dir / "best_model.pkl"
            joblib.dump(best_model, best_model_path)

    print(
        f"Best model: {best_name} (R2={best_r2:.3f}) "
        f"saved to {best_model_path}"
    )


if __name__ == "__main__":
    train()