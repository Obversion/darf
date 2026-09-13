import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RidgeEstimator:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.models = {}
        self.drws = {}
        self.r2_scores = {}  # node -> R² of the ridge fit of that node on its parents

    def fit(self, data_df: pd.DataFrame, graph, column_mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        reverse_map = {v: k for k, v in column_mapping.items()}
        drw = {}
        r2_scores = {}
        for node in graph.get_all_nodes():
            actual_node = column_mapping.get(node)
            if actual_node is None or actual_node not in data_df.columns:
                logger.warning(f"Node '{node}' has no valid column mapping; skipping DRW.")
                continue
            parents = graph.get_parents(node)
            if not parents:
                # Roots have no parents → R² undefined / treat as 0 (full intrinsic)
                r2_scores[node] = 0.0
                continue
            actual_parents = []
            for p in parents:
                actual_p = column_mapping.get(p)
                if actual_p is not None and actual_p in data_df.columns:
                    actual_parents.append(actual_p)
                else:
                    logger.warning(f"Parent '{p}' has no valid column mapping; skipping.")
            if not actual_parents:
                logger.warning(f"No valid parents for node '{node}'; skipping DRW.")
                r2_scores[node] = 0.0
                continue

            y_node = data_df[actual_node].values.ravel()
            X_parents = data_df[actual_parents].values

            ridge = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            ridge.fit(X_parents, y_node)

            coeffs = ridge.coef_
            logger.info(f"DRW coefficients for {node} (parents {actual_parents}): {dict(zip(parents, coeffs))}")

            abs_coeffs = np.abs(coeffs)
            if abs_coeffs.sum() > 0:
                drw[node] = {parents[i]: abs_coeffs[i] / abs_coeffs.sum()
                             for i in range(len(parents))}
            else:
                # Fallback: if all coefficients zero, use equal weights
                drw[node] = {p: 1.0 / len(parents) for p in parents}
                logger.warning(f"All coefficients zero for {node}; using equal DRW.")
            self.models[node] = ridge

            # Compute R² of the ridge fit (fraction of variance explained by parents)
            try:
                r2 = float(ridge.score(X_parents, y_node))
                # Clamp to [0, 1] in case of numerical issues
                r2 = max(0.0, min(1.0, r2))
            except Exception:
                r2 = 0.0
            r2_scores[node] = r2
            logger.info(f"R² for node {node}: {r2:.4f}")

        self.drws = drw
        self.r2_scores = r2_scores
        logger.info(f"DRW estimation complete for {len(drw)} nodes")
        logger.info(f"DRW dictionary: {drw}")
        logger.info(f"R² scores: {r2_scores}")
        return drw

    def get_drw(self, node: str) -> Dict[str, float]:
        return self.drws.get(node, {})

    def get_r2(self, node: str) -> float:
        return self.r2_scores.get(node, 0.0)

    def get_all_r2(self) -> Dict[str, float]:
        return dict(self.r2_scores)
