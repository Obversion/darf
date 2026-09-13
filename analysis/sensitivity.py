import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from joblib import Parallel, delayed
from config.config import DWRFConfigV4
from models.factory import ModelFactory
from evaluation.metrics import compute_metrics
from graph.ridge_estimator import RidgeEstimator
from graph.domain_prior import DomainPriorCalculator, apply_sampling_smoothing
import logging

logger = logging.getLogger(__name__)

def _stable_seed(base, value, offset=0):
    """Process-stable seed from a numeric parameter (does not use salted hash())."""
    try:
        scaled = int(round(abs(float(value)) * 1_000_000.0))
    except (TypeError, ValueError):
        scaled = 0
    return int(base) + int(offset) + scaled



class SensitivityAnalyzer:
    """Sensitivity analysis for graph structure and hyperparameters (α, λ)."""

    def __init__(self, config: DWRFConfigV4):
        self.config = config

    def _make_model(self, alg, feature_weights, feature_names, num_classes, seed=None):
        """Deterministic model for sensitivity: single-process trees + fixed seed."""
        kwargs = dict(
            feature_weights=feature_weights,
            num_classes=num_classes,
            feature_names=feature_names,
            n_jobs=1,
        )
        if seed is not None:
            kwargs['random_state'] = int(seed)
        return ModelFactory.create_model(self.config, alg, **kwargs)


    # ------------------------------------------------------------------
    # Graph sensitivity (unchanged logic)
    # ------------------------------------------------------------------
    def run_graph_sensitivity(self, data_dict: Dict) -> Dict[str, Any]:
        """Compare performance across different graphs."""
        results = {}
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        dpw_results = data_dict['dpw_results']
        feature_names = data_dict['feature_names']
        problem_type = data_dict['problem_type']
        n_runs = data_dict.get('n_runs', self.config.evaluation.n_runs)
        alg_types = data_dict.get('alg_types', ['bagging', 'boosting'])
        mape_threshold = data_dict.get('mape_threshold', None)
        num_classes = self.config.problem.num_classes if self.config.problem.type == "classification" else None

        for graph_name, dpw in dpw_results.items():
            for alg in alg_types:
                metrics_rf = []
                metrics_dwrf = []
                for run_i in range(n_runs):
                    seed = self.config.evaluation.random_seed_base + run_i
                    np.random.seed(seed)

                    model_rf = self._make_model(alg, None, feature_names, num_classes, seed=seed)
                    model_rf.fit(X_train, y_train)
                    y_pred_rf = model_rf.predict(X_test)

                    model_dwrf = self._make_model(alg, dpw, feature_names, num_classes, seed=seed + 17)
                    model_dwrf.fit(X_train, y_train)
                    y_pred_dwrf = model_dwrf.predict(X_test)

                    metrics_rf.append(compute_metrics(
                        y_test, y_pred_rf, problem_type, mape_threshold=mape_threshold))
                    metrics_dwrf.append(compute_metrics(
                        y_test, y_pred_dwrf, problem_type, mape_threshold=mape_threshold))

                avg_rf = {k: np.mean([m[k] for m in metrics_rf]) for k in metrics_rf[0]}
                avg_dwrf = {k: np.mean([m[k] for m in metrics_dwrf]) for k in metrics_dwrf[0]}
                key = f"{graph_name}_{alg}"
                results[key] = {'rf': avg_rf, 'dwrf': avg_dwrf}

        return results

    # ------------------------------------------------------------------
    # λ (ridge regularisation) sensitivity – FULL re-estimation
    # ------------------------------------------------------------------
    def run_lambda_sensitivity(self, data_dict: Dict) -> Dict[float, Dict]:
        """
        True λ-sensitivity: for every candidate ridge alpha (λ)

          1. Re-estimate Domain Relative Weights (DRW) with Ridge(alpha=λ)
          2. Re-compute Domain Prior Weights (DPW) with the new DRWs
             (and the intrinsic-influence / R² logic)
          3. Train a DWRF model with the fresh DPW and evaluate on the test set

        Returns a dict  λ → metrics.
        """
        # Obtain list of λ values
        if hasattr(self.config.ridge, 'lambda_sensitivity'):
            lambda_values = self.config.ridge.lambda_sensitivity.get('values', [])
        else:
            lambda_values = self.config.ridge.get('lambda_sensitivity', {}).get('values', [])

        if not lambda_values:
            logger.warning("No lambda_sensitivity values configured; skipping.")
            return {}

        X_train = data_dict.get('X_train')
        y_train = data_dict.get('y_train')
        X_test = data_dict.get('X_test')
        y_test = data_dict.get('y_test')
        feature_names = data_dict.get('feature_names', [])
        problem_type = data_dict.get('problem_type', 'regression')
        mape_threshold = data_dict.get('mape_threshold', None)
        alg = data_dict.get('alg', 'bagging')
        num_classes = (self.config.problem.num_classes
                       if self.config.problem.type == "classification" else None)

        # Objects required for re-estimation
        full_df_scaled = data_dict.get('full_df_scaled')
        graphs = data_dict.get('graphs') or {}
        target_col = data_dict.get('target_col', self.config.features.target)

        # Fall back to the pre-computed DPW if we cannot re-estimate
        can_reestimate = (full_df_scaled is not None and len(graphs) > 0)
        if not can_reestimate:
            logger.warning(
                "full_df_scaled / graphs missing from data_dict – "
                "λ-sensitivity will reuse the original DPW (incomplete).")

        # Use the first configured graph for the sensitivity sweep
        if graphs:
            graph_name = list(graphs.keys())[0]
            graph = graphs[graph_name]
        else:
            graph_name, graph = None, None

        # Build a simple column mapping (node name → column name)
        column_mapping = {}
        if graph is not None and full_df_scaled is not None:
            for node in graph.get_all_nodes():
                if node in full_df_scaled.columns:
                    column_mapping[node] = node
                else:
                    matches = [c for c in full_df_scaled.columns if c.lower() == node.lower()]
                    if matches:
                        column_mapping[node] = matches[0]
                    elif node.lower() == target_col.lower():
                        column_mapping[node] = target_col

        results = {}
        for lam in lambda_values:
            logger.info(f"λ-sensitivity: estimating DRW/DPW with λ={lam}")

            if can_reestimate and graph is not None:
                # 1. Re-estimate DRWs under the new regularisation strength
                ridge = RidgeEstimator(alpha=float(lam))
                drw = ridge.fit(full_df_scaled, graph, column_mapping)
                r2_scores = ridge.get_all_r2()

                # 2. Re-compute DPWs (with intrinsic influence)
                calc = DomainPriorCalculator(
                    min_intrinsic_influence=self.config.domain_prior.min_intrinsic_influence,
                    use_r_squared_intrinsic=self.config.domain_prior.use_r_squared_intrinsic,
                    fixed_intrinsic_influence=self.config.domain_prior.fixed_intrinsic_influence,
                )
                try:
                    raw_dpw = calc.compute(
                        graph, drw, target_col, feature_names,
                        column_mapping, r2_scores=r2_scores)
                    # Keep only model features and renormalise
                    dpw = {f: raw_dpw.get(f, 0.0) for f in feature_names}
                    total = sum(dpw.values())
                    if total > 0:
                        dpw = {f: w / total for f, w in dpw.items()}
                    else:
                        dpw = {f: 1.0 / len(feature_names) for f in feature_names}
                    alpha = getattr(self.config.sampling, 'smoothing_alpha', 0.2)
                    min_p = getattr(self.config.sampling, 'min_sampling_prob', 0.0)
                    dpw = apply_sampling_smoothing(dpw, alpha=alpha, min_prob=min_p)
                except Exception as e:
                    logger.error(f"DPW computation failed for λ={lam}: {e}")
                    # Fall back to uniform
                    dpw = {f: 1.0 / len(feature_names) for f in feature_names}
            else:
                # Incomplete path – reuse the DPW that was already computed
                dpw = data_dict.get('dpw') or {
                    f: 1.0 / len(feature_names) for f in feature_names
                }

            # 3. Train & evaluate with FIXED seed so charts are repeatable
            #    Seed depends only on base seed + λ (not on prior experiment RNG state)
            seed_base = int(getattr(self.config.evaluation, 'random_seed_base', 42))
            # stable integer from λ value
            seed = _stable_seed(seed_base, lam, offset=0)
            np.random.seed(seed)

            model = self._make_model(alg, dpw, feature_names, num_classes, seed=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(
                y_test, y_pred, problem_type, mape_threshold=mape_threshold)
            results[lam] = metrics
            logger.info(f"λ={lam} (seed={seed}) → MAE={metrics.get('MAE', float('nan')):.4f}, "
                        f"MAPE={metrics.get('MAPE', float('nan')):.4f}")

        return results

    # ------------------------------------------------------------------
    # α (smoothing) sensitivity – already correct
    # ------------------------------------------------------------------
    def run_alpha_sensitivity(self, data_dict: Dict) -> Dict[float, Dict]:
        """Run sensitivity for the DPW smoothing parameter α."""
        if hasattr(self.config.sampling, 'alpha_sensitivity'):
            alpha_values = self.config.sampling.alpha_sensitivity.get('values', [])
        else:
            alpha_values = self.config.sampling.get('alpha_sensitivity', {}).get('values', [])

        results = {}
        X_train = data_dict.get('X_train')
        y_train = data_dict.get('y_train')
        X_test = data_dict.get('X_test')
        y_test = data_dict.get('y_test')
        dpw = data_dict.get('dpw', {})
        alg = data_dict.get('alg', 'bagging')
        problem_type = data_dict.get('problem_type', 'regression')
        mape_threshold = data_dict.get('mape_threshold', None)
        num_classes = (self.config.problem.num_classes
                       if self.config.problem.type == "classification" else None)

        for alpha in alpha_values:
            n_features = len(dpw)
            if n_features > 0:
                feature_names = list(dpw.keys())
                weights = list(dpw.values())
                smoothed = [(1 - alpha) * w + alpha / n_features for w in weights]
                weighted_dpw = dict(zip(feature_names, smoothed))
            else:
                weighted_dpw = dpw

            # Fixed seed per α so the smoothing chart is stable across pipeline runs
            seed_base = int(getattr(self.config.evaluation, 'random_seed_base', 42))
            seed = _stable_seed(seed_base, alpha, offset=10_000_000)
            np.random.seed(seed)

            model = self._make_model(
                alg, weighted_dpw,
                list(dpw.keys()) if dpw else None,
                num_classes, seed=seed,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(
                y_test, y_pred, problem_type, mape_threshold=mape_threshold)
            results[alpha] = metrics
            logger.info(f"α={alpha} (seed={seed}) → MAE={metrics.get('MAE', float('nan')):.4f}")

        return results
