# evaluation/metrics.py
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, problem_type='regression', multi_class='ovr',
                    mape_threshold=None):
    """
    Compute metrics based on problem type.

    MAPE uses a "drop near-zero y" rule when mape_threshold > 0:
      - Only points with |y_true| > mape_threshold are used for MAPE.
      - mape_threshold must be in the **same units as y_true / y_pred**.
        If targets are StandardScaler-scaled, inverse-transform before calling
        this function (runner does this) so the floor is in physical units.
      - MAE, RMSE, and R2 always use the full dataset.

    If the floor excludes every point (e.g. threshold larger than max |y|),
    MAPE falls back to the epsilon-guarded full-set MAPE and a warning is logged
    so charts never lose the MAPE series.
    """
    if problem_type == 'regression':
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        epsilon = 1e-8

        def _mape(yt, yp):
            return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), epsilon))) * 100.0)

        result = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        if mape_threshold is not None and float(mape_threshold) > 0:
            floor = float(mape_threshold)
            mask = np.abs(y_true) > floor
            n_kept = int(mask.sum())
            n_total = int(len(y_true))
            pct_kept = (n_kept / n_total) * 100.0 if n_total else 0.0

            if n_kept > 0:
                mape = _mape(y_true[mask], y_pred[mask])
            else:
                # Floor too aggressive for this scale — keep MAPE visible
                logger.warning(
                    "mape_threshold=%.4g excluded all %d points "
                    "(max |y|=%.4g). Falling back to full-set MAPE. "
                    "Use a smaller floor in the same units as y, or ensure "
                    "y is inverse-scaled to physical units before metrics.",
                    floor, n_total, float(np.nanmax(np.abs(y_true))) if n_total else 0.0,
                )
                mape = _mape(y_true, y_pred)
                pct_kept = 100.0
                n_kept = n_total

            result['MAPE'] = mape
            result['MAPE_Min_Y'] = floor
            result['Pct_Above_Min_Y'] = pct_kept
            result['MAPE_N_Used'] = n_kept
            result['MAPE_N_Total'] = n_total
        else:
            result['MAPE'] = _mape(y_true, y_pred)

        return result

    elif problem_type == 'classification':
        y_pred_labels = np.round(y_pred).astype(int) if np.asarray(y_pred).dtype.kind in 'fi' else y_pred
        y_true = np.asarray(y_true).astype(int)
        accuracy = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred, multi_class=multi_class, average='macro')
        except Exception:
            auc = 0.0
        return {
            'accuracy': accuracy, 'f1_macro': f1,
            'precision_macro': precision, 'recall_macro': recall, 'auc_roc': auc,
        }
    else:
        raise ValueError("problem_type must be 'regression' or 'classification'")


def rank_biserial_correlation(x, y, higher_is_better=False):
    """Alias target for imports."""
    """
    Rank-biserial correlation (effect size for paired comparisons).
    Positive r means x is better than y.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = len(x)
    greater = np.sum(x > y)
    less = np.sum(x < y)
    ties = np.sum(x == y)
    cles = (greater + 0.5 * ties) / n
    if not higher_is_better:
        cles = 1 - cles
    r = 2 * cles - 1
    return r


def compute_common_language_effect_size(x, y, higher_is_better=False):
    """Compute CLES."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    greater = np.sum(x > y)
    ties = np.sum(x == y)
    n = len(x)
    cles = (greater + 0.5 * ties) / n
    if not higher_is_better:
        cles = 1 - cles
    return cles


def interpret_effect_size(r):
    if abs(r) >= 0.5:
        return "Large"
    elif abs(r) >= 0.3:
        return "Medium"
    elif abs(r) >= 0.1:
        return "Small"
    else:
        return "Negligible"


class EvaluationSummary:
    """Store and compare evaluation results."""

    def __init__(self, results: Dict[str, Dict[str, float]], problem_type: str):
        self.results = results
        self.problem_type = problem_type

    def compare_with(self, other: 'EvaluationSummary', metric: str = 'MAE', alpha: float = 0.05):
        """Compare two summaries (self vs other) on a metric using Wilcoxon test and effect size."""
        pass


def compute_rank_biserial_correlation(x, y, higher_is_better=False):
    return rank_biserial_correlation(x, y, higher_is_better=higher_is_better)
