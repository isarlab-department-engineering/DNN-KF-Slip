from joblib import load
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import numpy as np

def predict(input: np.array) -> float:

    model = load('e86eb65e_MLPR_c50e1006.joblib')
    best_slip=model.predict([input])[0]
    return best_slip

