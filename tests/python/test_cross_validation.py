import xgboost as xgb
from sklearn.datasets import make_regression
import numpy as np


class TestCrossValidation:
    def test_cross_validation_reg(self):
        rng = np.random.RandomState(1994)
        X, y = make_regression(n_samples=1000, n_features=10, random_state=rng)
        Xy = xgb.DMatrix(X, y)

        models, evals_log = xgb.training.kfold_cross_validation(
            {"tree_method": "hist"}, Xy
        )
