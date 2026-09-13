import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

from config.config import DWRFConfigV4, CVType

logger = logging.getLogger(__name__)

class PVDataLoader:
    """Load and preprocess data for DWRF with CV support."""

    def __init__(self, config: DWRFConfigV4):
        self.cfg = config
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_encoders = {}
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.folds = []
        self.test_indices = None  # Store test indices for temporal analysis

    def load(self) -> Dict:
        """Load and preprocess data."""
        logger.info("Loading data from %s", self.cfg.dataset.path)
        df = pd.read_csv(self.cfg.dataset.path)

        # Handle missing values
        df = self._handle_missing(df)

        # Encode categorical features (if any)
        df = self._encode_categorical(df)

        # Create lag features if specified
        df = self._create_lag_features(df)

        # Create cyclical features if specified
        df = self._create_cyclical_features(df)

        # Split features and target
        feature_cols = self.cfg.features.columns
        target_col = self.cfg.features.target

        X = df[feature_cols].values
        y = df[target_col].values

        # For classification, discretize target if needed
        if self.cfg.problem.type == "classification":
            y = self._discretize_target(y)

        # Scale features (scale y for regression)
        X_scaled = self.scaler_X.fit_transform(X)
        if self.cfg.problem.type == "regression":
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            y_scaled = y  # no scaling for classification

        self.data = {
            "X": X_scaled,
            "y": y_scaled,
            "X_raw": X,
            "y_raw": y,
            "feature_names": feature_cols,
            "target_name": target_col,
            "df": df
        }

        # Create CV folds
        self._create_folds(X_scaled, y_scaled)

        # Set train/val/test from first fold or from chronological split
        cv_type = self._get_cv_type()
        if cv_type == CVType.CHRONOLOGICAL:
            fold = self.folds[0]
            self.X_train, self.y_train = fold["train"]
            self.X_val, self.y_val = fold["val"]
            self.X_test, self.y_test = fold["test"]
        else:
            # Use first fold for train/val/test
            fold = self.folds[0]
            self.X_train, self.y_train = fold["train"]
            self.X_val, self.y_val = fold["val"]
            self.X_test, self.y_test = fold["test"]

        return self.data

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.cfg.dataset.missing_strategy
        if strategy == "drop":
            df = df.dropna()
        elif strategy == "mean":
            df = df.fillna(df.mean())
        elif strategy == "median":
            df = df.fillna(df.median())
        elif strategy == "most_frequent":
            df = df.fillna(df.mode().iloc[0])
        elif strategy == "constant":
            df = df.fillna(self.cfg.dataset.missing_fill_value)
        else:
            raise ValueError(f"Unknown missing strategy: {strategy}")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns (object dtype) using LabelEncoder."""
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.feature_encoders[col] = le
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lags = self.cfg.features.lags
        if not lags:
            return df
        # Sort by datetime if available
        if self.cfg.dataset.datetime_column:
            df = df.sort_values(self.cfg.dataset.datetime_column)
        for lag in lags:
            for col in self.cfg.features.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # Drop rows with NaN from shifts
        df = df.dropna()
        return df

    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert cyclical features (e.g., Hour, Month) to sin/cos."""
        for col in self.cfg.features.cyclical_features:
            if col in df.columns:
                if col == "Hour":
                    max_val = 24
                elif col == "Month":
                    max_val = 12
                else:
                    continue
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        return df

    def _discretize_target(self, y: np.ndarray) -> np.ndarray:
        """For classification, we may need to discretize continuous target."""
        return y

    def _get_cv_type(self) -> CVType:
        """Get CV type as enum, handling both string and enum values."""
        cv_type = self.cfg.dataset.cv_type
        if isinstance(cv_type, CVType):
            return cv_type
        elif isinstance(cv_type, str):
            # Convert string to enum
            try:
                return CVType(cv_type)
            except ValueError:
                # Try lowercase matching
                for enum_member in CVType:
                    if enum_member.value == cv_type.lower():
                        return enum_member
                raise ValueError(f"Unknown CV type string: {cv_type}")
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")

    def _create_folds(self, X: np.ndarray, y: np.ndarray):
        """Create folds based on cv_type."""
        cv_type = self._get_cv_type()
        if cv_type == CVType.CHRONOLOGICAL:
            self._create_chronological_folds(X, y)
        elif cv_type == CVType.BLOCKED:
            self._create_blocked_folds(X, y)
        elif cv_type == CVType.ROLLING:
            self._create_rolling_folds(X, y)
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")

    def _create_chronological_folds(self, X, y):
        """Single train/val/test split."""
        tr_ratio = self.cfg.dataset.train_ratio
        val_ratio = self.cfg.dataset.val_ratio
        n = len(X)
        train_end = int(tr_ratio * n)
        val_end = int((tr_ratio + val_ratio) * n)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        self.folds = [{"train": (X_train, y_train),
                       "val": (X_val, y_val),
                       "test": (X_test, y_test)}]
        # Store test indices for temporal analysis
        self.test_indices = slice(val_end, n)

    def _create_blocked_folds(self, X, y):
        """Blocked cross-validation: non-overlapping blocks."""
        n_folds = self.cfg.dataset.n_folds
        n = len(X)
        block_size = n // n_folds
        folds = []
        test_indices_list = []
        for i in range(n_folds):
            test_start = i * block_size
            test_end = (i + 1) * block_size if i < n_folds - 1 else n
            # Use all previous blocks as train, with validation from end of train
            train_val_end = test_start
            val_start = int(0.8 * train_val_end)
            X_train, y_train = X[:val_start], y[:val_start]
            X_val, y_val = X[val_start:train_val_end], y[val_start:train_val_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            folds.append({"train": (X_train, y_train),
                          "val": (X_val, y_val),
                          "test": (X_test, y_test)})
            test_indices_list.append(slice(test_start, test_end))
        self.folds = folds
        # Store first fold test indices
        if folds:
            self.test_indices = test_indices_list[0]

    def _create_rolling_folds(self, X, y):
        """Rolling time-series CV: expanding window."""
        n_folds = self.cfg.dataset.n_folds
        test_window = self.cfg.dataset.test_window
        step = self.cfg.dataset.step_size
        n = len(X)
        folds = []
        # Start with initial train size
        train_size = n - n_folds * test_window
        if train_size <= 0:
            train_size = n // 2
        for i in range(n_folds):
            train_end = train_size + i * step
            test_start = train_end
            test_end = test_start + test_window
            if test_end > n:
                break
            # Use a validation set from the end of training
            val_start = int(0.8 * train_end)
            X_train, y_train = X[:val_start], y[:val_start]
            X_val, y_val = X[val_start:train_end], y[val_start:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            folds.append({"train": (X_train, y_train),
                          "val": (X_val, y_val),
                          "test": (X_test, y_test)})
        self.folds = folds
        # Store first fold test indices
        if folds:
            self.test_indices = slice(test_start, test_end)

    def get_train_val_test(self, fold_idx=0):
        """Return train, val, test for a given fold."""
        fold = self.folds[fold_idx]
        return fold["train"], fold["val"], fold["test"]
    
    def get_test_indices(self):
        """Return test indices for temporal analysis."""
        return self.test_indices