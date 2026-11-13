from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "params.yaml"

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    data = fetch_california_housing(as_frame=True)
    features = data.frame.drop("MedHouseVal", axis=1)
    target = data.frame["MedHouseVal"]
    return train_test_split(
        features,
        target,
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )