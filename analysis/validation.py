"""Validation analysis for cross-validation strategies."""
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import KFold, TimeSeriesSplit
from evaluation.metrics import compute_metrics

class ValidationAnalyzer:
    """
    Analyze different cross-validation strategies.
    
    Supports:
    - Chronological (train/val/test split)
    - Blocked cross-validation (non-overlapping blocks)
    - Rolling time-series cross-validation (expanding window)
    """
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.cv_type = config.dataset.cv_type
        
    def validate_blocked_cv(self, X: np.ndarray, y: np.ndarray, 
                             model, n_folds: int = 5) -> Dict[str, List[float]]:
        """
        Perform blocked cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model instance with fit() and predict() methods
            n_folds: Number of folds
            
        Returns:
            Dictionary with metrics per fold
        """
        n = len(X)
        block_size = n // n_folds
        metrics = {'MAE': [], 'RMSE': [], 'R2': []}
        
        for i in range(n_folds):
            test_start = i * block_size
            test_end = (i + 1) * block_size if i < n_folds - 1 else n
            
            # Train on all previous blocks
            train_idx = list(range(0, test_start))
            test_idx = list(range(test_start, test_end))
            
            if len(train_idx) == 0:
                continue
                
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Compute metrics
            fold_metrics = compute_metrics(y_test, y_pred, self.config.problem.type)
            for k, v in fold_metrics.items():
                if k in metrics:
                    metrics[k].append(v)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        return avg_metrics
    
    def validate_rolling_cv(self, X: np.ndarray, y: np.ndarray,
                            model, n_folds: int = 5, 
                            test_window: int = 24) -> Dict[str, List[float]]:
        """
        Perform rolling time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model instance
            n_folds: Number of folds
            test_window: Size of test window
            
        Returns:
            Dictionary with metrics per fold
        """
        n = len(X)
        metrics = {'MAE': [], 'RMSE': [], 'R2': []}
        
        # Initial train size
        train_size = n - n_folds * test_window
        if train_size <= 0:
            train_size = n // 2
        
        for i in range(n_folds):
            train_end = train_size + i * test_window
            test_start = train_end
            test_end = min(test_start + test_window, n)
            
            if test_start >= n:
                break
                
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Compute metrics
            fold_metrics = compute_metrics(y_test, y_pred, self.config.problem.type)
            for k, v in fold_metrics.items():
                if k in metrics:
                    metrics[k].append(v)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
        return avg_metrics
    
    def compare_cv_strategies(self, X: np.ndarray, y: np.ndarray,
                              model_rf, model_dwrf) -> Dict[str, Dict]:
        """
        Compare performance across CV strategies.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_rf: Random Forest model (baseline)
            model_dwrf: DWRF model
            
        Returns:
            Dictionary with results for each strategy
        """
        results = {}
        
        if self.cv_type == "blocked":
            results['blocked'] = {
                'rf': self.validate_blocked_cv(X, y, model_rf, self.config.dataset.n_folds),
                'dwrf': self.validate_blocked_cv(X, y, model_dwrf, self.config.dataset.n_folds)
            }
        elif self.cv_type == "rolling":
            results['rolling'] = {
                'rf': self.validate_rolling_cv(X, y, model_rf, 
                                              self.config.dataset.n_folds,
                                              self.config.dataset.test_window),
                'dwrf': self.validate_rolling_cv(X, y, model_dwrf,
                                                self.config.dataset.n_folds,
                                                self.config.dataset.test_window)
            }
        else:
            # Chronological - use standard train/val/test split from loader
            results['chronological'] = {
                'rf': None,  # Will be filled by runner
                'dwrf': None
            }
        
        return results
    
    def get_validation_summary(self, results: Dict) -> str:
        """Generate a summary string of validation results."""
        summary = []
        summary.append("=" * 60)
        summary.append("CROSS-VALIDATION SUMMARY")
        summary.append("=" * 60)
        
        for strategy, data in results.items():
            summary.append(f"\nStrategy: {strategy.upper()}")
            if data['rf']:
                summary.append("  RF:")
                for k, v in data['rf'].items():
                    summary.append(f"    {k}: {v:.4f}")
            if data['dwrf']:
                summary.append("  DWRF:")
                for k, v in data['dwrf'].items():
                    summary.append(f"    {k}: {v:.4f}")
            # Compute improvement if both exist
            if data['rf'] and data['dwrf']:
                for k in data['rf']:
                    if k in data['dwrf']:
                        improvement = (data['rf'][k] - data['dwrf'][k]) / data['rf'][k] * 100
                        if 'MAE' in k or 'RMSE' in k or 'MAPE' in k:
                            # Lower is better
                            imp_str = f"{improvement:.2f}% improvement"
                        else:
                            # Higher is better (R2, accuracy)
                            imp_str = f"{-improvement:.2f}% improvement"
                        summary.append(f"    Improvement in {k}: {imp_str}")
        
        return "\n".join(summary)