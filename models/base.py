"""Base classes for weighted trees (not heavily used; we rely on sklearn)."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

class BaseWeightedTree:
    """Wrapper for weighted tree (not needed if we use sample weights)."""
    pass