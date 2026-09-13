# Unit tests for classification

import numpy as np
from sklearn.datasets import make_classification
from config.config import load_config, DWRFConfigV4
from models.factory import ModelFactory

def test_classification_bagging():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    config = load_config('config/config.yaml')
    config.problem.type = 'classification'
    config.problem.num_classes = 2
    # Use bagging
    config.algorithm.experiment_type = 'bagging'
    # Create model with no feature weights
    model = ModelFactory.create_model(config, 'bagging', feature_weights=None, num_classes=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]
    assert np.all((preds == 0) | (preds == 1))

def test_classification_boosting():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    config = load_config('config/config.yaml')
    config.problem.type = 'classification'
    config.problem.num_classes = 2
    config.algorithm.experiment_type = 'boosting'
    model = ModelFactory.create_model(config, 'boosting', feature_weights=None, num_classes=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape[0] == X.shape[0]

if __name__ == '__main__':
    test_classification_bagging()
    test_classification_boosting()
    print("Classification tests passed.")