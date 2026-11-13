 For logging, seeding, and reproducibility.
 
import numpy as np
import random
import os
import joblib
def set_seed(seed):
 np.random.seed(seed)
 random.seed(seed)
 os.environ["PYTHONHASHSEED"] = str(seed)
def save_model(model, path):
 joblib.dump(model, path)