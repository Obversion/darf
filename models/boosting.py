# models/boosting.py
"""Boosting models for DWRF V4.

- DomainWeightedXGBoost: standard XGBoost baseline (no DPW-guided sampling).
- DomainWeightedGradientBoosting: residual boosting with DPW-guided feature
  sampling (same mechanism as DomainWeightedRandomForest).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


def _resolve_feature_weights(
    feature_weights: Optional[Union[dict, list, np.ndarray]],
    n_features: int,
    feature_names: Optional[Sequence[str]],
) -> Optional[np.ndarray]:
    """Return a length-n_features probability vector, or None for uniform sampling."""
    if feature_weights is None:
        return None

    if isinstance(feature_weights, dict):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(n_features)]
        weights = np.array([float(feature_weights.get(f, 0.0)) for f in feature_names], dtype=float)
    else:
        weights = np.asarray(feature_weights, dtype=float).ravel()
        if len(weights) != n_features:
            logger.warning(
                "Feature weights length %d != n_features %d; using uniform.",
                len(weights), n_features,
            )
            return None

    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        logger.warning("Feature weights sum to zero/invalid; using uniform.")
        return None
    return weights / total


def _n_features_to_draw(max_features, n_features: int) -> int:
    if max_features == "sqrt":
        m = int(np.sqrt(n_features))
    elif max_features == "log2":
        m = int(np.log2(n_features)) + 1
    elif isinstance(max_features, int):
        m = max_features
    elif isinstance(max_features, float) and 0 < max_features <= 1:
        m = int(np.ceil(max_features * n_features))
    else:
        m = n_features
    return max(1, min(m, n_features))


class DomainWeightedGradientBoosting:
    """
    Gradient boosting with optional DPW-guided feature sampling.

    Each stage:
      1. residuals r = y - F  (regression)
      2. sample feature subset with p ∝ DPW (uniform if no weights)
      3. fit DecisionTree on (X[:, S], r) with optional row subsample
      4. F ← F + learning_rate * h
    """

    def __init__(
        self,
        config=None,
        feature_weights=None,
        num_classes=None,
        feature_names=None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        max_features="sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.config = config
        self.feature_weights = feature_weights
        self.num_classes = num_classes
        self.feature_names = feature_names
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.is_regression = (
            config is None or getattr(getattr(config, "problem", None), "type", "regression") == "regression"
        )
        self.trees_ = []
        self.feature_indices_ = []
        self.init_prediction_ = 0.0
        self.feature_selection_counts_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.trees_ = []
        self.feature_indices_ = []
        self.feature_selection_counts_ = np.zeros(n_features, dtype=float)

        weights = _resolve_feature_weights(self.feature_weights, n_features, self.feature_names)
        m = _n_features_to_draw(self.max_features, n_features)
        rng = np.random.RandomState(self.random_state)

        if self.is_regression:
            self.init_prediction_ = float(np.mean(y))
        else:
            # binary/multi simplified: mean of y as start (works for 0/1)
            self.init_prediction_ = float(np.mean(y))

        F = np.full(n_samples, self.init_prediction_, dtype=float)

        for t in range(self.n_estimators):
            residual = y - F

            if weights is not None:
                selected = rng.choice(n_features, size=m, replace=False, p=weights)
            else:
                selected = rng.choice(n_features, size=m, replace=False)
            selected = np.sort(selected)
            self.feature_selection_counts_[selected] += 1.0
            self.feature_indices_.append(selected)

            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * n_samples))
                row_idx = rng.choice(n_samples, size=n_sub, replace=False)
            else:
                row_idx = np.arange(n_samples)

            X_sub = X[np.ix_(row_idx, selected)]
            r_sub = residual[row_idx]

            if self.is_regression:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=None if self.random_state is None else int(self.random_state) + t,
                )
            else:
                tree = DecisionTreeRegressor(  # fit residuals even for classification start
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=None if self.random_state is None else int(self.random_state) + t,
                )
            tree.fit(X_sub, r_sub)
            self.trees_.append(tree)

            # update F on all samples
            h = tree.predict(X[:, selected])
            F = F + self.learning_rate * h

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        F = np.full(n, self.init_prediction_, dtype=float)
        for tree, selected in zip(self.trees_, self.feature_indices_):
            F = F + self.learning_rate * tree.predict(X[:, selected])
        return F

    def get_feature_selection_frequencies(self, feature_names=None):
        if self.feature_selection_counts_ is None or self.n_features_ is None:
            return {}
        names = feature_names or self.feature_names
        if names is None:
            names = [f"f{i}" for i in range(self.n_features_)]
        total = float(self.feature_selection_counts_.sum())
        if total <= 0:
            return {n: 0.0 for n in names}
        # frequency relative to n_estimators (how often selected)
        denom = float(self.n_estimators) if self.n_estimators else 1.0
        return {names[i]: float(self.feature_selection_counts_[i] / denom)
                for i in range(min(len(names), self.n_features_))}


class DomainWeightedXGBoost:
    """Standard XGBoost baseline — does not apply DPW-guided feature sampling."""

    def __init__(self, feature_weights=None, num_classes=None, **params):
        self.feature_weights = feature_weights  # ignored for training
        self.num_classes = num_classes
        self.params = dict(params)
        self.model_ = None
        self.is_regression = True

    def fit(self, X, y):
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost is required for the XGBoost baseline") from e

        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_estimators = int(self.params.get("n_estimators", 100))
        max_depth = int(self.params.get("max_depth", 6))
        learning_rate = float(self.params.get("learning_rate", 0.1))
        subsample = float(self.params.get("subsample", 0.8))
        colsample_bytree = float(self.params.get("colsample_bytree", 0.8))
        n_jobs = int(self.params.get("n_jobs", -1))
        objective = self.params.get("objective", "reg:squarederror")
        reg_alpha = float(self.params.get("reg_alpha", 0))
        reg_lambda = float(self.params.get("reg_lambda", 1))
        min_child_weight = float(self.params.get("min_child_weight", 1))
        gamma = float(self.params.get("gamma", 0))

        self.model_ = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            objective=objective,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            verbosity=0,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))
