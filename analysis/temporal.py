import numpy as np
from typing import Dict, Optional
from evaluation.metrics import compute_metrics


class TemporalAnalyzer:
    """Temporal performance analysis for RF, XGBoost, and domain-weighted experimental."""

    def __init__(self, config):
        self.config = config
        temporal = config.evaluation.temporal
        if hasattr(temporal, 'time_buckets'):
            self.time_buckets = temporal.time_buckets
            self.enabled = temporal.enabled
        else:
            self.time_buckets = temporal.get('time_buckets', {
                "morning": [6, 11],
                "noon": [11, 13],
                "afternoon": [14, 18]
            })
            self.enabled = temporal.get('enabled', True)

    def _metrics_for_mask(self, y_true, preds: Dict[str, np.ndarray], mask) -> Dict:
        mape_thr = getattr(self.config.evaluation, 'mape_threshold', None)
        out = {}
        yt = y_true[mask]
        for name, yp in preds.items():
            if yp is None:
                continue
            out[name] = compute_metrics(
                yt, yp[mask], self.config.problem.type, mape_threshold=mape_thr
            )
        return out

    def analyze_time_of_day(
        self,
        y_true: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_dwrf: np.ndarray,
        hours: np.ndarray,
        y_pred_xgb: Optional[np.ndarray] = None,
    ) -> Dict:
        """Analyze performance by time-of-day buckets for all available models."""
        y_true = np.asarray(y_true).ravel()
        hours = np.asarray(hours).ravel()
        preds = {
            'rf': np.asarray(y_pred_rf).ravel(),
            'dwrf': np.asarray(y_pred_dwrf).ravel(),
        }
        if y_pred_xgb is not None:
            preds['xgb'] = np.asarray(y_pred_xgb).ravel()

        results = {}
        for bucket_name, (start, end) in self.time_buckets.items():
            mask = (hours >= start) & (hours < end)
            if mask.sum() == 0:
                continue
            results[bucket_name] = self._metrics_for_mask(y_true, preds, mask)
        return results

    def analyze_hour_by_hour(
        self,
        y_true: np.ndarray,
        y_pred_rf: np.ndarray,
        y_pred_dwrf: np.ndarray,
        hours: np.ndarray,
        y_pred_xgb: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict]:
        """Analyze performance for each hour (0-23) for all available models."""
        y_true = np.asarray(y_true).ravel()
        hours = np.asarray(hours).ravel()
        preds = {
            'rf': np.asarray(y_pred_rf).ravel(),
            'dwrf': np.asarray(y_pred_dwrf).ravel(),
        }
        if y_pred_xgb is not None:
            preds['xgb'] = np.asarray(y_pred_xgb).ravel()

        results = {}
        for h in np.unique(hours):
            mask = hours == h
            if mask.sum() == 0:
                continue
            results[int(h)] = self._metrics_for_mask(y_true, preds, mask)
        return results
