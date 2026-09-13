# Unit tests for boosting

import numpy as np
from sklearn.datasets import make_regression
from config.config import load_config
from models.factory import ModelFactory

def test_boosting_regression():
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
    config = load_config('config/config.yaml')
    config.problem.type = 'regression'
    config.algorithm.experiment_type = 'boosting'
    model = ModelFactory.create_model(config, 'boosting', feature_weights=None)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all(np.isfinite(preds))

if __name__ == '__main__':
    test_boosting_regression()
    print("Boosting test passed.")