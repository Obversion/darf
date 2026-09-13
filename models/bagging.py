import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)

class DomainWeightedRandomForest:
    def __init__(self, config, feature_weights=None, num_classes=None, feature_names=None, **kwargs):
        self.config = config
        self.feature_weights = feature_weights   # dict or None
        self.num_classes = num_classes
        self.feature_names = feature_names
        self.params = kwargs
        self.is_regression = (config.problem.type == "regression")
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.trees = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_estimators = self.params.get('n_estimators', 100)
        max_depth = self.params.get('max_depth', None)
        min_samples_split = self.params.get('min_samples_split', 2)
        min_samples_leaf = self.params.get('min_samples_leaf', 1)
        max_features = self.params.get('max_features', 'sqrt')
        class_weight = self.params.get('class_weight', None)

        # Convert feature_weights (dict) to array aligned with feature order
        if self.feature_weights is not None and isinstance(self.feature_weights, dict):
            if self.feature_names is None:
                # Try to get from X if it's a DataFrame
                if hasattr(X, 'columns'):
                    self.feature_names = list(X.columns)
                else:
                    self.feature_names = [f'f{i}' for i in range(n_features)]
            weights = np.array([self.feature_weights.get(f, 0.0) for f in self.feature_names])
            if weights.sum() == 0:
                weights = np.ones(n_features) / n_features
            else:
                weights = weights / weights.sum()
        elif self.feature_weights is not None and isinstance(self.feature_weights, (list, np.ndarray)):
            weights = np.array(self.feature_weights)
            if len(weights) != n_features:
                logger.warning(f"Feature weights length mismatch; using uniform.")
                weights = np.ones(n_features) / n_features
            else:
                weights = weights / weights.sum()
        else:
            weights = None

        # Determine m (number of features per tree)
        if max_features == 'sqrt':
            m = int(np.sqrt(n_features))
        elif max_features == 'log2':
            m = int(np.log2(n_features))
        elif isinstance(max_features, int):
            m = max_features
        else:
            m = n_features
        m = max(1, min(m, n_features))

        def _train_tree(seed):
            rng = np.random.RandomState(int(seed))
            idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            if weights is not None:
                selected = rng.choice(n_features, size=m, replace=False, p=weights)
            else:
                selected = rng.choice(n_features, size=m, replace=False)
            if self.is_regression:
                tree = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=seed
                )
            else:
                tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=seed
                )
            tree.fit(X_boot[:, selected], y_boot)
            return (tree, selected)

        rs = self.params.get('random_state', None)
        rng = np.random.RandomState(rs)
        seeds = rng.randint(0, 2**31, size=n_estimators)
        n_jobs = self.n_jobs if self.n_jobs != -1 else -1
        self.trees = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_train_tree)(seed) for seed in seeds
        )
        return self

    def predict(self, X):
        if self.trees is None:
            raise ValueError("Model not fitted.")
        n_jobs = self.n_jobs if self.n_jobs != -1 else -1
        preds = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(lambda t: t[0].predict(X[:, t[1]]))(t) for t in self.trees
        )
        preds = np.array(preds)
        if self.is_regression:
            return preds.mean(axis=0)
        else:
            from scipy.stats import mode
            return mode(preds, axis=0)[0].ravel()

    def predict_proba(self, X):
        if self.is_regression:
            raise ValueError("predict_proba not for regression.")
        if self.trees is None:
            raise ValueError("Model not fitted.")
        n_jobs = self.n_jobs if self.n_jobs != -1 else -1
        probs = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(lambda t: t[0].predict_proba(X[:, t[1]]))(t) for t in self.trees
        )
        return np.array(probs).mean(axis=0)
    
    def get_feature_selection_frequencies(self, feature_names=None):
        """
        Return a dict of feature name -> selection frequency (0..1).
        """
        if self.trees is None:
            return {}
        n_features = self.trees[0][1].shape[0] if self.trees else 0
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(n_features)]
        counts = {f: 0 for f in feature_names}
        for tree, selected in self.trees:
            for idx in selected:
                counts[feature_names[idx]] += 1
        total = len(self.trees)
        return {f: counts[f] / total for f in feature_names}