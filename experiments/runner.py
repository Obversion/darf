# # experiments/runner.py
# import numpy as np
# import pandas as pd
# import logging
# import json
# import time
# from pathlib import Path
# from typing import Dict, Any
# from joblib import Parallel, delayed
# from scipy.stats import wilcoxon
# import warnings

# from config.config import DWRFConfigV4, AlgorithmType
# from data.loader import PVDataLoader
# from graph.multi_graph import MultiGraphManager
# from graph.ridge_estimator import RidgeEstimator
# from graph.domain_prior import DomainPriorCalculator, apply_sampling_smoothing
# from models.factory import ModelFactory
# from evaluation.metrics import compute_metrics, compute_rank_biserial_correlation
# from analysis.sensitivity import SensitivityAnalyzer
# from analysis.temporal import TemporalAnalyzer
# from visualization.plots import DWRFVisualizer
# from export.onnx_exporter import ONNXExporter
# from sklearn.preprocessing import StandardScaler

# logger = logging.getLogger(__name__)

# def safe_wilcoxon(x, y, **kwargs):
#     """
#     Safe wrapper for Wilcoxon signed-rank test that handles ties and zero variance.
    
#     Returns:
#         (statistic, p_value) or (0, 1.0) if test fails
#     """
#     x = np.asarray(x).ravel()
#     y = np.asarray(y).ravel()
    
#     # Check if we have enough data
#     if len(x) < 2 or len(y) < 2:
#         return 0, 1.0
    
#     # Check if all values are identical (zero variance)
#     if np.std(x) == 0 and np.std(y) == 0:
#         return 0, 1.0
    
#     # Check if all differences are zero (perfect correlation)
#     diff = x - y
#     if np.all(diff == 0):
#         return 0, 1.0
    
#     # Check for constant differences (all positive or all negative)
#     if np.all(diff > 0) or np.all(diff < 0):
#         n = len(x)
#         r_plus = n * (n + 1) / 2
#         return r_plus, 2 * (1 - 0.5)
    
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", RuntimeWarning)
#             non_zero_mask = diff != 0
#             if np.sum(non_zero_mask) < 2:
#                 return 0, 1.0
#             stat, p = wilcoxon(x[non_zero_mask], y[non_zero_mask], **kwargs)
#             return stat, p
#     except (ValueError, RuntimeWarning) as e:
#         logger.debug(f"Wilcoxon test failed: {e}")
#         return 0, 1.0

# class DWRFRunner:
#     def __init__(self, config: DWRFConfigV4):
#         self.cfg = config
#         self.data_loader = PVDataLoader(config)
#         self.graph_manager = MultiGraphManager(config)
#         self.sensitivity_analyzer = SensitivityAnalyzer(config)
#         self.temporal_analyzer = TemporalAnalyzer(config)
#         self.visualizer = None
#         self.results = {
#             'dpw': {},
#             'dpw_raw': {},
#             'drw': {},
#             'models': {},
#             'sensitivity': {},
#             'temporal': {},
#             'statistical_tests': {},
#             'summary': {},
#             'feature_frequencies': {},
#             'timing': {},
#             'predictions': {}
#         }
#         self.output_dir = Path(config.output.dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         logging.basicConfig(
#             level=getattr(logging, config.logging.level),
#             filename=config.logging.log_file,
#             filemode='w',
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         )
#         if config.logging.console_output:
#             console = logging.StreamHandler()
#             console.setLevel(getattr(logging, config.logging.level))
#             logging.getLogger('').addHandler(console)

#     def run(self) -> Dict:
#         logger.info("Starting DARF V4 experiment: %s", self.cfg.experiment.name)

#         # Get MAPE threshold from config
#         mape_threshold = self.cfg.evaluation.mape_threshold
#         logger.info(
#             f"MAPE near-zero floor (mape_threshold) set to: {mape_threshold} "
#             f"(physical units after inverse-scaling y)"
#         )

#         def _inverse_y(y):
#             """Map y / predictions back to physical units for metrics & plots."""
#             if self.cfg.problem.type != 'regression':
#                 return y
#             scaler = getattr(self.data_loader, 'scaler_y', None)
#             if scaler is None:
#                 return y
#             y = np.asarray(y, dtype=float).ravel()
#             return scaler.inverse_transform(y.reshape(-1, 1)).ravel()


#         # 1. Load data - measure preprocessing time
#         start_time = time.time()
#         data = self.data_loader.load()
#         X_train, y_train = self.data_loader.X_train, self.data_loader.y_train
#         X_val, y_val = self.data_loader.X_val, self.data_loader.y_val
#         X_test, y_test = self.data_loader.X_test, self.data_loader.y_test
#         feature_names = data['feature_names']
#         full_df = data['df']
#         preprocessing_time = time.time() - start_time
#         logger.info(f"Preprocessing time: {preprocessing_time:.3f} seconds")

#         scaler_full = StandardScaler()
#         full_df_scaled = full_df.copy()
#         numeric_cols = full_df.select_dtypes(include=[np.number]).columns
#         full_df_scaled[numeric_cols] = scaler_full.fit_transform(full_df[numeric_cols])

#         logger.info(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
#         logger.info(f"Feature names: {feature_names}")
#         logger.info(f"Target column: {self.cfg.features.target}")
#         logger.info(f"Full DataFrame columns: {list(full_df_scaled.columns)}")

#         # 2. Build graphs (original, cause → effect)
#         graphs = self.graph_manager.build_all_graphs()
#         dpw_results = {}
#         dpw_raw_results = {}
#         drw_results = {}

#         for graph_name, graph in graphs.items():
#             logger.info("=" * 60)
#             logger.info(f"Processing graph: {graph_name}")
#             logger.info("=" * 60)

#             column_mapping = {}
#             for node in graph.get_all_nodes():
#                 if node in full_df_scaled.columns:
#                     column_mapping[node] = node
#                     logger.info(f"Exact match: '{node}' -> '{node}'")
#                 else:
#                     matches = [col for col in full_df_scaled.columns if col.lower() == node.lower()]
#                     if matches:
#                         column_mapping[node] = matches[0]
#                         logger.info(f"Case-insensitive match: '{node}' -> '{matches[0]}'")
#                     else:
#                         if node.lower() == self.cfg.features.target.lower():
#                             column_mapping[node] = self.cfg.features.target
#                             logger.info(f"Target match: '{node}' -> '{self.cfg.features.target}'")
#                         else:
#                             logger.warning(f"No mapping found for graph node '{node}'")

#             missing_nodes = [n for n in graph.get_all_nodes() if n not in column_mapping]
#             if missing_nodes:
#                 logger.warning(f"Graph {graph_name} has nodes with no column mapping: {missing_nodes}")
#                 for node in missing_nodes:
#                     if node in full_df_scaled.columns:
#                         column_mapping[node] = node

#             logger.info(f"Estimating DRWs with alpha={self.cfg.ridge.alpha}...")
#             ridge = RidgeEstimator(alpha=self.cfg.ridge.alpha)
#             drw = ridge.fit(full_df_scaled, graph, column_mapping)
#             logger.info(f"DRW estimation complete for {len(drw)} nodes")
            
#             drw_results[graph_name] = drw
#             for node, parents in drw.items():
#                 logger.debug(f"DRW for {node}: {parents}")

#             target_col = self.cfg.features.target
#             logger.info(f"Computing DPWs for target: {target_col}")
#             calc = DomainPriorCalculator(
#                 min_intrinsic_influence=self.cfg.domain_prior.min_intrinsic_influence,
#                 use_r_squared_intrinsic=self.cfg.domain_prior.use_r_squared_intrinsic,
#                 fixed_intrinsic_influence=self.cfg.domain_prior.fixed_intrinsic_influence
#             )
#             try:
#                 r2_scores = ridge.get_all_r2() if hasattr(ridge, 'get_all_r2') else {}
#                 raw_dpw = calc.compute(
#                     graph, drw, target_col, feature_names, column_mapping,
#                     r2_scores=r2_scores,
#                 )
#                 logger.info(f"Raw DPW for {graph_name}: {raw_dpw}")
#                 logger.info(f"R² used for intrinsic γ: {r2_scores}")

#                 dpw_filtered = {f: raw_dpw.get(f, 0.0) for f in feature_names}
#                 total = sum(dpw_filtered.values())
#                 if total == 0:
#                     raise RuntimeError(f"DPW for graph {graph_name} has zero sum on input features.")
#                 dpw_filtered = {f: w / total for f, w in dpw_filtered.items()}

#                 # Eq. (21): training sampling probabilities (does not overwrite raw DPW plot)
#                 alpha = getattr(self.cfg.sampling, 'smoothing_alpha', 0.2)
#                 min_p = getattr(self.cfg.sampling, 'min_sampling_prob', 0.0)
#                 dpw_train = apply_sampling_smoothing(dpw_filtered, alpha=alpha, min_prob=min_p)

#                 dpw_raw_results[graph_name] = raw_dpw
#                 dpw_results[graph_name] = dpw_train
#                 logger.info(f"Filtered DPW (pre-smooth) for {graph_name}: {dpw_filtered}")
#                 logger.info(f"Training sampling P (α={alpha}) for {graph_name}: {dpw_train}")

#             except RuntimeError as e:
#                 logger.error(f"DPW computation failed: {e}")
#                 logger.error("Skipping this graph.")
#                 continue

#         self.results['dpw'] = dpw_results
#         self.results['dpw_raw'] = dpw_raw_results
#         self.results['drw'] = drw_results

#         if not dpw_results:
#             raise RuntimeError("No DPW results computed for any graph. Check your graph configuration.")

#         # 3. Domain-weighted experimental mode from enabled flags
#         #    Baselines are ALWAYS standard RF and standard XGBoost.
#         #    Experimental model is DARF (bagging) and/or DW-GB (boosting).
#         bag_enabled = bool(getattr(self.cfg.algorithm.bagging, 'enabled', True))
#         boost_enabled = bool(getattr(self.cfg.algorithm.boosting, 'enabled', False))
#         exp_type = self.cfg.algorithm.experiment_type

#         # Resolve which domain-weighted families to run
#         dw_families = []
#         if exp_type == AlgorithmType.BAGGING or (bag_enabled and not boost_enabled):
#             dw_families = ['bagging']
#         elif exp_type == AlgorithmType.BOOSTING or (boost_enabled and not bag_enabled):
#             dw_families = ['boosting']
#         elif exp_type == AlgorithmType.COMPARISON or (bag_enabled and boost_enabled):
#             dw_families = []
#             if bag_enabled:
#                 dw_families.append('bagging')
#             if boost_enabled:
#                 dw_families.append('boosting')
#             if not dw_families:
#                 # comparison with both disabled: fall back to both
#                 dw_families = ['bagging', 'boosting']
#         else:
#             dw_families = ['bagging'] if bag_enabled else (['boosting'] if boost_enabled else ['bagging'])

#         logger.info(
#             f"Baselines: RF + XGBoost always | Domain-weighted experimental: {dw_families} "
#             f"(bagging.enabled={bag_enabled}, boosting.enabled={boost_enabled}, "
#             f"experiment_type={exp_type})"
#         )
#         alg_types = list(dw_families)  # used downstream for export / sensitivity

#         # 4. Parallel experiments
#         n_runs = self.cfg.evaluation.n_runs
#         num_classes = self.cfg.problem.num_classes if self.cfg.problem.type == "classification" else None

#         def _run_single_exp(graph_name, dpw, dw_family, run_idx):
#             """
#             Always train RF + XGBoost baselines.
#             Train domain-weighted experimental for dw_family:
#               bagging  → DARF
#               boosting → DomainWeightedGradientBoosting
#             """
#             try:
#                 seed = self.cfg.evaluation.random_seed_base + run_idx
#                 np.random.seed(seed)

#                 # ---- RF baseline (always) ----
#                 t0 = time.perf_counter()
#                 model_rf = ModelFactory.create_model(
#                     self.cfg, 'bagging',
#                     feature_weights=None,
#                     num_classes=num_classes,
#                     feature_names=feature_names,
#                 )
#                 model_rf.fit(X_train, y_train)
#                 rf_train = time.perf_counter() - t0
#                 t0 = time.perf_counter()
#                 y_pred_rf = model_rf.predict(X_test)
#                 rf_infer = time.perf_counter() - t0

#                 # ---- XGBoost baseline (always) ----
#                 t0 = time.perf_counter()
#                 model_xgb = ModelFactory.create_model(
#                     self.cfg, 'boosting',
#                     feature_weights=None,
#                     num_classes=num_classes,
#                     feature_names=feature_names,
#                 )
#                 model_xgb.fit(X_train, y_train)
#                 xgb_train = time.perf_counter() - t0
#                 t0 = time.perf_counter()
#                 y_pred_xgb = model_xgb.predict(X_test)
#                 xgb_infer = time.perf_counter() - t0

#                 # ---- Domain-weighted experimental ----
#                 t0 = time.perf_counter()
#                 model_dw = ModelFactory.create_model(
#                     self.cfg, dw_family,
#                     feature_weights=dpw,
#                     num_classes=num_classes,
#                     feature_names=feature_names,
#                 )
#                 model_dw.fit(X_train, y_train)
#                 dw_train = time.perf_counter() - t0
#                 t0 = time.perf_counter()
#                 y_pred_dw = model_dw.predict(X_test)
#                 dw_infer = time.perf_counter() - t0

#                 metrics_rf = compute_metrics(
#                     _inverse_y(y_test), _inverse_y(y_pred_rf),
#                     problem_type=self.cfg.problem.type,
#                     mape_threshold=mape_threshold,
#                 )
#                 metrics_xgb = compute_metrics(
#                     _inverse_y(y_test), _inverse_y(y_pred_xgb),
#                     problem_type=self.cfg.problem.type,
#                     mape_threshold=mape_threshold,
#                 )
#                 metrics_dw = compute_metrics(
#                     _inverse_y(y_test), _inverse_y(y_pred_dw),
#                     problem_type=self.cfg.problem.type,
#                     mape_threshold=mape_threshold,
#                 )

#                 return {
#                     'metrics_rf': metrics_rf,
#                     'metrics_xgb': metrics_xgb,
#                     'metrics_dw': metrics_dw,
#                     'rf_train': rf_train, 'rf_infer': rf_infer,
#                     'xgb_train': xgb_train, 'xgb_infer': xgb_infer,
#                     'dw_train': dw_train, 'dw_infer': dw_infer,
#                     'y_rf': y_pred_rf, 'y_xgb': y_pred_xgb, 'y_dw': y_pred_dw,
#                     'dw_family': dw_family,
#                 }
#             except Exception as e:
#                 logger.error(
#                     f"Error in run {run_idx} for {graph_name} {dw_family}: {e}",
#                     exc_info=True,
#                 )
#                 return None

#         task_list = []
#         for graph_name, dpw in dpw_results.items():
#             for dw_family in dw_families:
#                 for run_idx in range(n_runs):
#                     task_list.append((graph_name, dpw, dw_family, run_idx))

#         logger.info(f"Total tasks: {len(task_list)}")
#         n_jobs = self.cfg.algorithm.bagging.n_jobs if 'bagging' in dw_families else (
#             self.cfg.algorithm.boosting.n_jobs if 'boosting' in dw_families else 1
#         )

#         results_list = Parallel(n_jobs=n_jobs)(
#             delayed(_run_single_exp)(g, d, fam, r) for (g, d, fam, r) in task_list
#         )

#         valid_results = [r for r in results_list if r is not None]
#         logger.info(f"Valid results: {len(valid_results)} / {len(results_list)}")
#         if not valid_results:
#             logger.error("No valid results returned. Check model training or data.")
#             return self.results

#         # ---- Aggregate ----
#         aggregated = {}  # key = f"{graph}_{family}" → lists of metrics
#         train_times = {'RF': [], 'XGBoost': [], 'DARF': []}
#         inference_times = {'RF': [], 'XGBoost': [], 'DARF': []}
#         predictions_by_graph = {}
#         first_graph_name = list(dpw_results.keys())[0] if dpw_results else None

#         for (graph_name, dpw, dw_family, run_idx), result in zip(task_list, results_list):
#             if not result:
#                 continue
#             key = f"{graph_name}_{dw_family}"
#             if key not in aggregated:
#                 aggregated[key] = {'rf': [], 'xgb': [], 'dwrf': []}
#             aggregated[key]['rf'].append(result['metrics_rf'])
#             aggregated[key]['xgb'].append(result['metrics_xgb'])
#             aggregated[key]['dwrf'].append(result['metrics_dw'])

#             train_times['RF'].append(result['rf_train'])
#             inference_times['RF'].append(result['rf_infer'])
#             train_times['XGBoost'].append(result['xgb_train'])
#             inference_times['XGBoost'].append(result['xgb_infer'])
#             train_times['DARF'].append(result['dw_train'])
#             inference_times['DARF'].append(result['dw_infer'])

#             if graph_name == first_graph_name and dw_family == dw_families[0]:
#                 predictions_by_graph.setdefault('RF', []).append(result['y_rf'])
#                 predictions_by_graph.setdefault('XGBoost', []).append(result['y_xgb'])
#                 predictions_by_graph.setdefault('DARF', []).append(result['y_dw'])

#         self.results['predictions'] = {}
#         for name, preds in predictions_by_graph.items():
#             if preds:
#                 self.results['predictions'][name] = _inverse_y(np.mean(preds, axis=0))
#                 logger.info(f"{name} predictions averaged over {len(preds)} runs (physical units)")

#         avg_times = {
#             'Preprocessing': preprocessing_time,
#             'Training': {
#                 k: (float(np.mean(v)) if v and any(t > 0 for t in v) else 0.0)
#                 for k, v in train_times.items()
#             },
#             'Inference': {
#                 k: (float(np.mean(v)) if v and any(t > 0 for t in v) else 0.0)
#                 for k, v in inference_times.items()
#             },
#         }
#         self.results['timing'] = avg_times
#         logger.info(f"Average timing: {avg_times}")

#         model_results = {}
#         for key, vals in aggregated.items():
#             avg_rf = {k: float(np.nanmean([m[k] for m in vals['rf']])) for k in vals['rf'][0]}
#             avg_xgb = {k: float(np.nanmean([m[k] for m in vals['xgb']])) for k in vals['xgb'][0]}
#             avg_dw = {k: float(np.nanmean([m[k] for m in vals['dwrf']])) for k in vals['dwrf'][0]}
#             # Plot/visualizer keys: *_rf, *_dwrf; XGBoost stored as boosting_rf when family is boosting
#             # Also always store explicit xgb under bagging key pattern for viz
#             model_results[f"{key}_rf"] = avg_rf
#             model_results[f"{key}_dwrf"] = avg_dw
#             model_results[f"{key}_xgb"] = avg_xgb

#         self.results['models'] = model_results
#         logger.info(f"Model result keys: {list(model_results.keys())}")

#         # Statistical tests: domain-weighted vs RF and vs XGBoost
#         base_metric_names = ['MAE', 'RMSE', 'MAPE', 'R2']
#         stat_results = {}
#         for key, vals in aggregated.items():
#             for m in base_metric_names:
#                 if m not in vals['rf'][0]:
#                     continue
#                 higher_better = m not in ['MAE', 'RMSE', 'MAPE']
#                 # DW vs RF
#                 rf_vals = [x[m] for x in vals['rf']]
#                 dw_vals = [x[m] for x in vals['dwrf']]
#                 try:
#                     _, p = safe_wilcoxon(dw_vals, rf_vals)
#                 except Exception:
#                     p = 1.0
#                 try:
#                     r = compute_rank_biserial_correlation(dw_vals, rf_vals, higher_is_better=higher_better)
#                 except Exception:
#                     r = 0.0
#                 stat_results[f"{key}_bagging_{m}"] = {
#                     'p_value': float(p), 'effect_size': float(r),
#                     'comparison': 'dw_vs_rf',
#                 }
#                 # DW vs XGBoost
#                 xgb_vals = [x[m] for x in vals['xgb']]
#                 try:
#                     _, p2 = safe_wilcoxon(dw_vals, xgb_vals)
#                 except Exception:
#                     p2 = 1.0
#                 try:
#                     r2 = compute_rank_biserial_correlation(dw_vals, xgb_vals, higher_is_better=higher_better)
#                 except Exception:
#                     r2 = 0.0
#                 # Use boosting tag so plots can show DW vs XGB comparison label
#                 stat_results[f"{key}_boosting_{m}"] = {
#                     'p_value': float(p2), 'effect_size': float(r2),
#                     'comparison': 'dw_vs_xgb',
#                 }
#         self.results['statistical_tests'] = stat_results

#         # Viz keys: primary graph + primary DW family
#         first_graph = list(dpw_results.keys())[0]
#         primary_family = dw_families[0]
#         dpw_for_viz = dpw_results[first_graph]
#         viz_alg = primary_family

#         self.data_loader.data['y_test'] = _inverse_y(y_test)
#         self.data_loader.data['y_pred_rf'] = self.results['predictions'].get('RF')
#         self.data_loader.data['y_pred_xgb'] = self.results['predictions'].get('XGBoost')
#         self.data_loader.data['y_pred_dwrf'] = self.results['predictions'].get('DARF')

#         rf_key = f"{first_graph}_{primary_family}_rf"
#         dwrf_key = f"{first_graph}_{primary_family}_dwrf"
#         xgb_key = f"{first_graph}_{primary_family}_xgb"
#         if rf_key not in model_results:
#             # fallback any _rf
#             for k in model_results:
#                 if k.endswith('_rf'):
#                     rf_key = k
#                     break
#         if dwrf_key not in model_results:
#             for k in model_results:
#                 if k.endswith('_dwrf'):
#                     dwrf_key = k
#                     break
#         if xgb_key not in model_results:
#             for k in model_results:
#                 if k.endswith('_xgb'):
#                     xgb_key = k
#                     break

#         self.data_loader.data['rf_metrics'] = model_results.get(rf_key, {})
#         self.data_loader.data['dwrf_metrics'] = model_results.get(dwrf_key, {})
#         self.data_loader.data['xgb_metrics'] = model_results.get(xgb_key, {})

#         # Store MAPE threshold for visualizer
#         self.data_loader.data['mape_threshold'] = mape_threshold

#         # Collect feature frequencies: RF baseline + domain-weighted experimental
#         if True:
#             model_rf_base = ModelFactory.create_model(
#                 self.cfg, 'bagging',
#                 feature_weights=None,
#                 num_classes=num_classes,
#                 feature_names=feature_names
#             )
#             model_rf_base.fit(X_train, y_train)
#             if hasattr(model_rf_base, 'get_feature_selection_frequencies'):
#                 self.results['feature_frequencies']['RF_baseline'] = \
#                     model_rf_base.get_feature_selection_frequencies(feature_names)

#             for graph_name, dpw in dpw_results.items():
#                 fam = dw_families[0] if dw_families else 'bagging'
#                 model = ModelFactory.create_model(
#                     self.cfg, fam,
#                     feature_weights=dpw,
#                     num_classes=num_classes,
#                     feature_names=feature_names
#                 )
#                 model.fit(X_train, y_train)
#                 if hasattr(model, 'get_feature_selection_frequencies'):
#                     freq = model.get_feature_selection_frequencies(feature_names)
#                     self.results['feature_frequencies'][graph_name] = freq

#         # 5. Sensitivity analysis
#         if self.cfg.ridge.lambda_sensitivity.get('enabled', False) or \
#            self.cfg.sampling.alpha_sensitivity.get('enabled', False):
#             data_dict = {
#                 'X_train': X_train,
#                 'y_train': y_train,
#                 'X_test': X_test,
#                 'y_test': y_test,
#                 'dpw_results': dpw_results,
#                 'feature_names': feature_names,
#                 'problem_type': self.cfg.problem.type,
#                 'n_runs': n_runs,
#                 'alg_types': alg_types,
#                 'dpw': dpw_for_viz,
#                 'alg': viz_alg
#             }
#             self.run_sensitivity_analysis(data_dict)

#         # 6. Temporal analysis — RF + XGBoost baselines + domain-weighted experimental
#         if self.cfg.evaluation.temporal.enabled and 'Hour' in full_df.columns:
#             graph_name = first_graph
#             dpw = dpw_for_viz
#             dw_family = dw_families[0] if dw_families else 'bagging'

#             model_rf_temp = ModelFactory.create_model(
#                 self.cfg, 'bagging',
#                 feature_weights=None,
#                 num_classes=num_classes,
#                 feature_names=feature_names,
#             )
#             model_rf_temp.fit(X_train, y_train)
#             y_pred_rf_temp = model_rf_temp.predict(X_test)

#             model_xgb_temp = ModelFactory.create_model(
#                 self.cfg, 'boosting',
#                 feature_weights=None,
#                 num_classes=num_classes,
#                 feature_names=feature_names,
#             )
#             model_xgb_temp.fit(X_train, y_train)
#             y_pred_xgb_temp = model_xgb_temp.predict(X_test)

#             model_dwrf_temp = ModelFactory.create_model(
#                 self.cfg, dw_family,
#                 feature_weights=dpw,
#                 num_classes=num_classes,
#                 feature_names=feature_names,
#             )
#             model_dwrf_temp.fit(X_train, y_train)
#             y_pred_dwrf_temp = model_dwrf_temp.predict(X_test)

#             if hasattr(self.data_loader, 'test_indices'):
#                 hours_test = full_df.iloc[self.data_loader.test_indices]['Hour'].values
#             else:
#                 hours_test = full_df['Hour'].values[-len(X_test):]

#             # Metrics in physical units (same as main evaluation)
#             y_true_t = _inverse_y(y_test)
#             y_rf_t = _inverse_y(y_pred_rf_temp)
#             y_xgb_t = _inverse_y(y_pred_xgb_temp)
#             y_dw_t = _inverse_y(y_pred_dwrf_temp)

#             self.results['temporal'] = self.temporal_analyzer.analyze_time_of_day(
#                 y_true_t, y_rf_t, y_dw_t, hours_test, y_pred_xgb=y_xgb_t
#             )
#             hourly = self.temporal_analyzer.analyze_hour_by_hour(
#                 y_true_t, y_rf_t, y_dw_t, hours_test, y_pred_xgb=y_xgb_t
#             )
#             self.results['temporal']['hourly'] = hourly
#             logger.info("Temporal analysis completed (RF, XGBoost, experimental).")

#         # 7. Generate visualizations
#         if dpw_raw_results:
#             self.generate_visualizations(feature_names, dpw_raw_results, drw_results, self.cfg.graphs)
#         else:
#             logger.warning("No raw DPW results, skipping visualizations.")

#         # 8. Export
#         if self.cfg.export.enabled and 'bagging' in alg_types:
#             # Use the last trained DARF model for export
#             exporter = ONNXExporter(self.cfg)
#             # Re-train a DARF model for export
#             model_export = ModelFactory.create_model(
#                 self.cfg, 'bagging',
#                 feature_weights=dpw_for_viz,
#                 num_classes=num_classes,
#                 feature_names=feature_names
#             )
#             model_export.fit(X_train, y_train)
#             exporter.export(model_export, X_train[:10], feature_names)

#         # 9. Save results
#         self.save_results()
#         logger.info("Experiment completed successfully.")
#         return self.results

#     def run_sensitivity_analysis(self, data_dict):
#         if self.cfg.ridge.lambda_sensitivity.get('enabled', False):
#             self.results['sensitivity']['lambda'] = self.sensitivity_analyzer.run_lambda_sensitivity(data_dict)
#         if self.cfg.sampling.alpha_sensitivity.get('enabled', False):
#             self.results['sensitivity']['alpha'] = self.sensitivity_analyzer.run_alpha_sensitivity(data_dict)

#     def generate_visualizations(self, feature_names, dpw_raw_results, drw_results, all_graphs):
#         if not self.results['models']:
#             logger.warning("No model results to visualize. Skipping plots.")
#             return

#         first_graph = list(dpw_raw_results.keys())[0]
#         dpw = dpw_raw_results[first_graph]
#         models = self.results['models']

#         def _first_key(suffix):
#             # Prefer primary graph keys
#             preferred = [k for k in models if k.startswith(first_graph + '_') and k.endswith(suffix)]
#             if preferred:
#                 return preferred[0]
#             any_k = [k for k in models if k.endswith(suffix)]
#             return any_k[0] if any_k else None

#         rf_key = _first_key('_rf')
#         dwrf_key = _first_key('_dwrf')
#         xgb_key = _first_key('_xgb')
#         logger.info(f"Viz keys: rf={rf_key}, dwrf={dwrf_key}, xgb={xgb_key}")

#         self.visualizer = DWRFVisualizer(
#             config=self.cfg,
#             feature_names=feature_names,
#             dpw=dpw,
#             dpw_raw=dpw_raw_results,
#             drw=drw_results,
#             graph_edges=all_graphs,
#             rf_summary=models.get(rf_key, {}) if rf_key else {},
#             dwrf_summary=models.get(dwrf_key, {}) if dwrf_key else {},
#             xgb_summary=models.get(xgb_key, {}) if xgb_key else {},
#             multi_graph_results=models,
#             sensitivity_results=self.results.get('sensitivity', {}),
#             temporal_results=self.results.get('temporal', {}),
#             data=self.data_loader.data,
#             stat_results=self.results.get('statistical_tests', {}),
#             feature_selection_freq=self.results.get('feature_frequencies', {}),
#             timing_data=self.results.get('timing', {})
#         )
#         self.visualizer.generate_all()
#         logger.info("Visualizations generated.")

#     def save_results(self):
#         output_path = self.output_dir / 'results_summary.json'
#         with open(output_path, 'w') as f:
#             json.dump(self.results, f, indent=2, default=str)
#         logger.info("Results saved to %s", output_path)


# experiments/runner.py
import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any
from contextlib import contextmanager
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
import warnings

try:
    from threadpoolctl import threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:
    _HAS_THREADPOOLCTL = False

from config.config import DWRFConfigV4, AlgorithmType
from data.loader import PVDataLoader
from graph.multi_graph import MultiGraphManager
from graph.ridge_estimator import RidgeEstimator
from graph.domain_prior import DomainPriorCalculator, apply_sampling_smoothing
from models.factory import ModelFactory
from evaluation.metrics import compute_metrics, compute_rank_biserial_correlation
from analysis.sensitivity import SensitivityAnalyzer
from analysis.temporal import TemporalAnalyzer
from visualization.plots import DWRFVisualizer
from export.onnx_exporter import ONNXExporter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _limit_threads(n):
    """Limit BLAS/OpenMP threads during a fit so timings are reproducible."""
    if _HAS_THREADPOOLCTL:
        return threadpool_limits(limits=n)
    return contextmanager(lambda: (yield))()


def safe_wilcoxon(x, y, **kwargs):
    """
    Safe wrapper for Wilcoxon signed-rank test that handles ties and zero variance.

    Returns:
        (statistic, p_value) or (0, 1.0) if test fails
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if len(x) < 2 or len(y) < 2:
        return 0, 1.0

    if np.std(x) == 0 and np.std(y) == 0:
        return 0, 1.0

    diff = x - y
    if np.all(diff == 0):
        return 0, 1.0

    if np.all(diff > 0) or np.all(diff < 0):
        n = len(x)
        r_plus = n * (n + 1) / 2
        return r_plus, 2 * (1 - 0.5)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            non_zero_mask = diff != 0
            if np.sum(non_zero_mask) < 2:
                return 0, 1.0
            stat, p = wilcoxon(x[non_zero_mask], y[non_zero_mask], **kwargs)
            return stat, p
    except (ValueError, RuntimeWarning) as e:
        logger.debug(f"Wilcoxon test failed: {e}")
        return 0, 1.0


class DWRFRunner:
    def __init__(self, config: DWRFConfigV4):
        self.cfg = config
        self.data_loader = PVDataLoader(config)
        self.graph_manager = MultiGraphManager(config)
        self.sensitivity_analyzer = SensitivityAnalyzer(config)
        self.temporal_analyzer = TemporalAnalyzer(config)
        self.visualizer = None
        self.results = {
            'dpw': {},
            'dpw_raw': {},
            'drw': {},
            'models': {},
            'sensitivity': {},
            'temporal': {},
            'statistical_tests': {},
            'summary': {},
            'feature_frequencies': {},
            'timing': {},
            'predictions': {}
        }
        self.output_dir = Path(config.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            filename=config.logging.log_file,
            filemode='w',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if config.logging.console_output:
            console = logging.StreamHandler()
            console.setLevel(getattr(logging, config.logging.level))
            logging.getLogger('').addHandler(console)

    # ------------------------------------------------------------------
    # Reference timing: reproducible, dedicated, seeded measurement
    # ------------------------------------------------------------------
    def _measure_reference_timing(
        self,
        X_train, y_train, X_test,
        feature_names,
        dpw,
        dw_family,
        num_classes,
        n_repeats=5,
    ):
        """
        Dedicated timing pass used for the computational-times chart.

        Fixed seed + single-threaded + warm-up + median for reproducible
        numbers. Does not affect any metric, prediction, or export.
        """
        logger.info("Measuring reference timing (fixed seed, single-threaded, %d repeats)",
                    n_repeats)

        timing_seed = 12345

        def _make_and_time(family, weights):
            with _limit_threads(1):
                np.random.seed(timing_seed)

                t0 = time.perf_counter()
                c0 = time.process_time()
                model = ModelFactory.create_model(
                    self.cfg, family,
                    feature_weights=weights,
                    num_classes=num_classes,
                    feature_names=feature_names,
                )
                model.fit(X_train, y_train)
                train_wall = time.perf_counter() - t0
                train_cpu = time.process_time() - c0

                t0 = time.perf_counter()
                c0 = time.process_time()
                _ = model.predict(X_test)
                infer_wall = time.perf_counter() - t0
                infer_cpu = time.process_time() - c0

            return train_wall, infer_wall, train_cpu, infer_cpu

        # Warm-up (discarded)
        try:
            _make_and_time('bagging', None)
            _make_and_time('boosting', None)
            _make_and_time(dw_family, dpw)
            logger.debug("Reference-timing warm-up completed.")
        except Exception as e:
            logger.warning(f"Reference-timing warm-up failed: {e}")

        def _collect(family, weights):
            tw, iw, tc, ic = [], [], [], []
            for _ in range(n_repeats):
                try:
                    a, b, c, d = _make_and_time(family, weights)
                    tw.append(a); iw.append(b); tc.append(c); ic.append(d)
                except Exception as e:
                    logger.warning(f"Reference-timing repeat failed for {family}: {e}")
            return tw, iw, tc, ic

        rf_tw, rf_iw, rf_tc, rf_ic = _collect('bagging', None)
        xgb_tw, xgb_iw, xgb_tc, xgb_ic = _collect('boosting', None)
        dw_tw, dw_iw, dw_tc, dw_ic = _collect(dw_family, dpw)

        def _med(vals):
            vals = [float(v) for v in vals if v is not None and v > 0]
            return float(np.median(vals)) if vals else 0.0

        timing = {
            'Training': {
                'RF': _med(rf_tw),
                'XGBoost': _med(xgb_tw),
                'DARF': _med(dw_tw),
            },
            'Inference': {
                'RF': _med(rf_iw),
                'XGBoost': _med(xgb_iw),
                'DARF': _med(dw_iw),
            },
            'Training_CPU': {
                'RF': _med(rf_tc),
                'XGBoost': _med(xgb_tc),
                'DARF': _med(dw_tc),
            },
            'Inference_CPU': {
                'RF': _med(rf_ic),
                'XGBoost': _med(xgb_ic),
                'DARF': _med(dw_ic),
            },
        }

        logger.info(
            "Reference timing (median of %d repeats): "
            "train wall = %s | train CPU = %s | infer wall = %s | infer CPU = %s",
            n_repeats,
            timing['Training'], timing['Training_CPU'],
            timing['Inference'], timing['Inference_CPU'],
        )
        return timing

    # ------------------------------------------------------------------
    # Train-only scaling & DRW/DPW computation (leakage-free)
    # ------------------------------------------------------------------
    def _build_train_only_scaler(self, full_df, numeric_cols, train_indices):
        """
        Fit a StandardScaler on the training rows only, then transform every
        row in full_df using that fitted scaler.

        This prevents test-set statistics from entering the scaling used by
        DRW estimation and DPW computation.

        Returns:
            (full_df_scaled, scaler_full)
        """
        scaler_full = StandardScaler()

        if train_indices is None:
            logger.warning(
                "Train indices unavailable; falling back to fitting the "
                "scaler on the full dataset. This reintroduces test-set "
                "information into the scaling."
            )
            full_df_scaled = full_df.copy()
            full_df_scaled[numeric_cols] = scaler_full.fit_transform(
                full_df[numeric_cols]
            )
            return full_df_scaled, scaler_full

        train_rows = full_df.iloc[train_indices]
        scaler_full.fit(train_rows[numeric_cols])

        full_df_scaled = full_df.copy()
        full_df_scaled[numeric_cols] = scaler_full.transform(full_df[numeric_cols])
        return full_df_scaled, scaler_full

    def _resolve_train_indices(self, full_df):
        """
        Try several strategies to identify which rows of full_df belong to
        the training split.

        Order of preference:
          1. data_loader.train_indices (if the loader exposes it)
          2. data_loader.X_train.index (if X_train was derived from full_df)
          3. None (caller must decide what to do)
        """
        # 1. Loader-provided attribute
        ti = getattr(self.data_loader, 'train_indices', None)
        if ti is not None:
            ti = np.asarray(ti)
            if ti.dtype.kind in ('i', 'u') and ti.max() < len(full_df):
                logger.info(f"Using data_loader.train_indices ({len(ti)} rows).")
                return ti
            logger.warning("data_loader.train_indices present but out of range; ignoring.")

        # 2. X_train index
        X_train = getattr(self.data_loader, 'X_train', None)
        if X_train is not None and hasattr(X_train, 'index'):
            idx = X_train.index
            # If the index values are positions into full_df, use them directly
            try:
                idx_arr = np.asarray(idx)
                if idx_arr.dtype.kind in ('i', 'u') and idx_arr.max() < len(full_df):
                    logger.info(f"Using X_train.index ({len(idx_arr)} rows).")
                    return idx_arr
                # Otherwise, if full_df's index contains those values, use
                # get_indexer to map them to positions.
                pos = full_df.index.get_indexer(idx)
                if np.all(pos >= 0):
                    logger.info(
                        f"Using X_train.index mapped via full_df.index "
                        f"({len(pos)} rows)."
                    )
                    return pos
            except Exception as e:
                logger.debug(f"X_train.index resolution failed: {e}")

        logger.warning(
            "Could not resolve training-row indices; the caller should "
            "fall back to a leakage-safe default or warn the user."
        )
        return None

    def run(self) -> Dict:
        logger.info("Starting DARF V4 experiment: %s", self.cfg.experiment.name)

        mape_threshold = self.cfg.evaluation.mape_threshold
        logger.info(
            f"MAPE near-zero floor (mape_threshold) set to: {mape_threshold} "
            f"(physical units after inverse-scaling y)"
        )

        def _inverse_y(y):
            if self.cfg.problem.type != 'regression':
                return y
            scaler = getattr(self.data_loader, 'scaler_y', None)
            if scaler is None:
                return y
            y = np.asarray(y, dtype=float).ravel()
            return scaler.inverse_transform(y.reshape(-1, 1)).ravel()

        # 1. Load data - measure preprocessing time
        start_time = time.time()
        data = self.data_loader.load()
        X_train, y_train = self.data_loader.X_train, self.data_loader.y_train
        X_val, y_val = self.data_loader.X_val, self.data_loader.y_val
        X_test, y_test = self.data_loader.X_test, self.data_loader.y_test
        feature_names = data['feature_names']
        full_df = data['df']
        preprocessing_time = time.time() - start_time
        logger.info(f"Preprocessing time: {preprocessing_time:.3f} seconds")

        # ------------------------------------------------------------------
        # 1a. Resolve which rows of full_df belong to the training split.
        # ------------------------------------------------------------------
        train_indices = self._resolve_train_indices(full_df)

        # ------------------------------------------------------------------
        # 1b. Fit the scaler on training rows only, then transform all rows.
        #     This replaces the previous full-dataset scaling and is the
        #     first half of the leakage fix.
        # ------------------------------------------------------------------
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns
        full_df_scaled, scaler_full = self._build_train_only_scaler(
            full_df, numeric_cols, train_indices
        )

        logger.info(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Target column: {self.cfg.features.target}")
        logger.info(f"Full DataFrame columns: {list(full_df_scaled.columns)}")
        if train_indices is not None:
            logger.info(
                f"Train-only scaling: {len(train_indices)} rows used for "
                f"fit; {len(full_df)} rows transformed."
            )
        else:
            logger.warning(
                "Train-only scaling NOT applied (indices unavailable); "
                "scaler was fit on the full dataset."
            )

        # 2. Build graphs (original, cause → effect)
        graphs = self.graph_manager.build_all_graphs()
        dpw_results = {}
        dpw_raw_results = {}
        drw_results = {}

        for graph_name, graph in graphs.items():
            logger.info("=" * 60)
            logger.info(f"Processing graph: {graph_name}")
            logger.info("=" * 60)

            column_mapping = {}
            for node in graph.get_all_nodes():
                if node in full_df_scaled.columns:
                    column_mapping[node] = node
                    logger.info(f"Exact match: '{node}' -> '{node}'")
                else:
                    matches = [col for col in full_df_scaled.columns if col.lower() == node.lower()]
                    if matches:
                        column_mapping[node] = matches[0]
                        logger.info(f"Case-insensitive match: '{node}' -> '{matches[0]}'")
                    else:
                        if node.lower() == self.cfg.features.target.lower():
                            column_mapping[node] = self.cfg.features.target
                            logger.info(f"Target match: '{node}' -> '{self.cfg.features.target}'")
                        else:
                            logger.warning(f"No mapping found for graph node '{node}'")

            missing_nodes = [n for n in graph.get_all_nodes() if n not in column_mapping]
            if missing_nodes:
                logger.warning(f"Graph {graph_name} has nodes with no column mapping: {missing_nodes}")
                for node in missing_nodes:
                    if node in full_df_scaled.columns:
                        column_mapping[node] = node

            # ------------------------------------------------------------------
            # 2a. Fit the Ridge estimator on TRAINING ROWS ONLY.
            #     This is the second half of the leakage fix: DRW values
            #     (and therefore DPWs) no longer see test-set rows.
            # ------------------------------------------------------------------
            logger.info(f"Estimating DRWs with alpha={self.cfg.ridge.alpha}...")
            if train_indices is not None:
                train_df_scaled = full_df_scaled.iloc[train_indices]
                logger.info(
                    f"Ridge fit on training rows only "
                    f"({len(train_df_scaled)} of {len(full_df_scaled)} rows)."
                )
            else:
                train_df_scaled = full_df_scaled
                logger.warning(
                    "Ridge fit on full dataset (train indices unavailable). "
                    "DRWs will include test-set information."
                )

            ridge = RidgeEstimator(alpha=self.cfg.ridge.alpha)
            drw = ridge.fit(train_df_scaled, graph, column_mapping)
            logger.info(f"DRW estimation complete for {len(drw)} nodes")

            drw_results[graph_name] = drw
            for node, parents in drw.items():
                logger.debug(f"DRW for {node}: {parents}")

            target_col = self.cfg.features.target
            logger.info(f"Computing DPWs for target: {target_col}")
            calc = DomainPriorCalculator(
                min_intrinsic_influence=self.cfg.domain_prior.min_intrinsic_influence,
                use_r_squared_intrinsic=self.cfg.domain_prior.use_r_squared_intrinsic,
                fixed_intrinsic_influence=self.cfg.domain_prior.fixed_intrinsic_influence
            )
            try:
                r2_scores = ridge.get_all_r2() if hasattr(ridge, 'get_all_r2') else {}
                raw_dpw = calc.compute(
                    graph, drw, target_col, feature_names, column_mapping,
                    r2_scores=r2_scores,
                )
                logger.info(f"Raw DPW for {graph_name}: {raw_dpw}")
                logger.info(f"R² used for intrinsic γ: {r2_scores}")

                dpw_filtered = {f: raw_dpw.get(f, 0.0) for f in feature_names}
                total = sum(dpw_filtered.values())
                if total == 0:
                    raise RuntimeError(f"DPW for graph {graph_name} has zero sum on input features.")
                dpw_filtered = {f: w / total for f, w in dpw_filtered.items()}

                alpha = getattr(self.cfg.sampling, 'smoothing_alpha', 0.2)
                min_p = getattr(self.cfg.sampling, 'min_sampling_prob', 0.0)
                dpw_train = apply_sampling_smoothing(dpw_filtered, alpha=alpha, min_prob=min_p)

                dpw_raw_results[graph_name] = raw_dpw
                dpw_results[graph_name] = dpw_train
                logger.info(f"Filtered DPW (pre-smooth) for {graph_name}: {dpw_filtered}")
                logger.info(f"Training sampling P (α={alpha}) for {graph_name}: {dpw_train}")

            except RuntimeError as e:
                logger.error(f"DPW computation failed: {e}")
                logger.error("Skipping this graph.")
                continue

        self.results['dpw'] = dpw_results
        self.results['dpw_raw'] = dpw_raw_results
        self.results['drw'] = drw_results

        if not dpw_results:
            raise RuntimeError("No DPW results computed for any graph. Check your graph configuration.")

        # 3. Domain-weighted experimental mode from enabled flags
        bag_enabled = bool(getattr(self.cfg.algorithm.bagging, 'enabled', True))
        boost_enabled = bool(getattr(self.cfg.algorithm.boosting, 'enabled', False))
        exp_type = self.cfg.algorithm.experiment_type

        dw_families = []
        if exp_type == AlgorithmType.BAGGING or (bag_enabled and not boost_enabled):
            dw_families = ['bagging']
        elif exp_type == AlgorithmType.BOOSTING or (boost_enabled and not bag_enabled):
            dw_families = ['boosting']
        elif exp_type == AlgorithmType.COMPARISON or (bag_enabled and boost_enabled):
            dw_families = []
            if bag_enabled:
                dw_families.append('bagging')
            if boost_enabled:
                dw_families.append('boosting')
            if not dw_families:
                dw_families = ['bagging', 'boosting']
        else:
            dw_families = ['bagging'] if bag_enabled else (['boosting'] if boost_enabled else ['bagging'])

        logger.info(
            f"Baselines: RF + XGBoost always | Domain-weighted experimental: {dw_families} "
            f"(bagging.enabled={bag_enabled}, boosting.enabled={boost_enabled}, "
            f"experiment_type={exp_type})"
        )
        alg_types = list(dw_families)

        # 4. Parallel experiments
        n_runs = self.cfg.evaluation.n_runs
        num_classes = self.cfg.problem.num_classes if self.cfg.problem.type == "classification" else None

        def _run_single_exp(graph_name, dpw, dw_family, run_idx):
            try:
                seed = self.cfg.evaluation.random_seed_base + run_idx
                np.random.seed(seed)

                with _limit_threads(1):
                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    model_rf = ModelFactory.create_model(
                        self.cfg, 'bagging',
                        feature_weights=None,
                        num_classes=num_classes,
                        feature_names=feature_names,
                    )
                    model_rf.fit(X_train, y_train)
                    rf_train = time.perf_counter() - t0
                    rf_train_cpu = time.process_time() - c0

                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    y_pred_rf = model_rf.predict(X_test)
                    rf_infer = time.perf_counter() - t0
                    rf_infer_cpu = time.process_time() - c0

                with _limit_threads(1):
                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    model_xgb = ModelFactory.create_model(
                        self.cfg, 'boosting',
                        feature_weights=None,
                        num_classes=num_classes,
                        feature_names=feature_names,
                    )
                    model_xgb.fit(X_train, y_train)
                    xgb_train = time.perf_counter() - t0
                    xgb_train_cpu = time.process_time() - c0

                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    y_pred_xgb = model_xgb.predict(X_test)
                    xgb_infer = time.perf_counter() - t0
                    xgb_infer_cpu = time.process_time() - c0

                with _limit_threads(1):
                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    model_dw = ModelFactory.create_model(
                        self.cfg, dw_family,
                        feature_weights=dpw,
                        num_classes=num_classes,
                        feature_names=feature_names,
                    )
                    model_dw.fit(X_train, y_train)
                    dw_train = time.perf_counter() - t0
                    dw_train_cpu = time.process_time() - c0

                    t0 = time.perf_counter()
                    c0 = time.process_time()
                    y_pred_dw = model_dw.predict(X_test)
                    dw_infer = time.perf_counter() - t0
                    dw_infer_cpu = time.process_time() - c0

                metrics_rf = compute_metrics(
                    _inverse_y(y_test), _inverse_y(y_pred_rf),
                    problem_type=self.cfg.problem.type,
                    mape_threshold=mape_threshold,
                )
                metrics_xgb = compute_metrics(
                    _inverse_y(y_test), _inverse_y(y_pred_xgb),
                    problem_type=self.cfg.problem.type,
                    mape_threshold=mape_threshold,
                )
                metrics_dw = compute_metrics(
                    _inverse_y(y_test), _inverse_y(y_pred_dw),
                    problem_type=self.cfg.problem.type,
                    mape_threshold=mape_threshold,
                )

                return {
                    'metrics_rf': metrics_rf,
                    'metrics_xgb': metrics_xgb,
                    'metrics_dw': metrics_dw,
                    'rf_train': rf_train, 'rf_infer': rf_infer,
                    'rf_train_cpu': rf_train_cpu, 'rf_infer_cpu': rf_infer_cpu,
                    'xgb_train': xgb_train, 'xgb_infer': xgb_infer,
                    'xgb_train_cpu': xgb_train_cpu, 'xgb_infer_cpu': xgb_infer_cpu,
                    'dw_train': dw_train, 'dw_infer': dw_infer,
                    'dw_train_cpu': dw_train_cpu, 'dw_infer_cpu': dw_infer_cpu,
                    'y_rf': y_pred_rf, 'y_xgb': y_pred_xgb, 'y_dw': y_pred_dw,
                    'dw_family': dw_family,
                }
            except Exception as e:
                logger.error(
                    f"Error in run {run_idx} for {graph_name} {dw_family}: {e}",
                    exc_info=True,
                )
                return None

        task_list = []
        for graph_name, dpw in dpw_results.items():
            for dw_family in dw_families:
                for run_idx in range(n_runs):
                    task_list.append((graph_name, dpw, dw_family, run_idx))

        logger.info(f"Total tasks: {len(task_list)}")
        n_jobs = self.cfg.algorithm.bagging.n_jobs if 'bagging' in dw_families else (
            self.cfg.algorithm.boosting.n_jobs if 'boosting' in dw_families else 1
        )

        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_exp)(g, d, fam, r) for (g, d, fam, r) in task_list
        )

        valid_results = [r for r in results_list if r is not None]
        logger.info(f"Valid results: {len(valid_results)} / {len(results_list)}")
        if not valid_results:
            logger.error("No valid results returned. Check model training or data.")
            return self.results

        # ---- Aggregate ----
        aggregated = {}
        train_times = {'RF': [], 'XGBoost': [], 'DARF': []}
        inference_times = {'RF': [], 'XGBoost': [], 'DARF': []}
        train_times_cpu = {'RF': [], 'XGBoost': [], 'DARF': []}
        inference_times_cpu = {'RF': [], 'XGBoost': [], 'DARF': []}
        predictions_by_graph = {}
        first_graph_name = list(dpw_results.keys())[0] if dpw_results else None

        for (graph_name, dpw, dw_family, run_idx), result in zip(task_list, results_list):
            if not result:
                continue
            key = f"{graph_name}_{dw_family}"
            if key not in aggregated:
                aggregated[key] = {'rf': [], 'xgb': [], 'dwrf': []}
            aggregated[key]['rf'].append(result['metrics_rf'])
            aggregated[key]['xgb'].append(result['metrics_xgb'])
            aggregated[key]['dwrf'].append(result['metrics_dw'])

            train_times['RF'].append(result['rf_train'])
            inference_times['RF'].append(result['rf_infer'])
            train_times['XGBoost'].append(result['xgb_train'])
            inference_times['XGBoost'].append(result['xgb_infer'])
            train_times['DARF'].append(result['dw_train'])
            inference_times['DARF'].append(result['dw_infer'])

            train_times_cpu['RF'].append(result['rf_train_cpu'])
            inference_times_cpu['RF'].append(result['rf_infer_cpu'])
            train_times_cpu['XGBoost'].append(result['xgb_train_cpu'])
            inference_times_cpu['XGBoost'].append(result['xgb_infer_cpu'])
            train_times_cpu['DARF'].append(result['dw_train_cpu'])
            inference_times_cpu['DARF'].append(result['dw_infer_cpu'])

            if graph_name == first_graph_name and dw_family == dw_families[0]:
                predictions_by_graph.setdefault('RF', []).append(result['y_rf'])
                predictions_by_graph.setdefault('XGBoost', []).append(result['y_xgb'])
                predictions_by_graph.setdefault('DARF', []).append(result['y_dw'])

        self.results['predictions'] = {}
        for name, preds in predictions_by_graph.items():
            if preds:
                self.results['predictions'][name] = _inverse_y(np.mean(preds, axis=0))
                logger.info(f"{name} predictions averaged over {len(preds)} runs (physical units)")

        def _median_or_zero(vals):
            vals = [float(v) for v in vals if v is not None and v > 0]
            return float(np.median(vals)) if vals else 0.0

        avg_times = {
            'Preprocessing': preprocessing_time,
            'Training': {k: _median_or_zero(v) for k, v in train_times.items()},
            'Inference': {k: _median_or_zero(v) for k, v in inference_times.items()},
            'Training_CPU': {k: _median_or_zero(v) for k, v in train_times_cpu.items()},
            'Inference_CPU': {k: _median_or_zero(v) for k, v in inference_times_cpu.items()},
        }
        self.results['timing'] = avg_times
        logger.info(f"Provisional timing (median over metric runs): {avg_times}")

        model_results = {}
        for key, vals in aggregated.items():
            avg_rf = {k: float(np.nanmean([m[k] for m in vals['rf']])) for k in vals['rf'][0]}
            avg_xgb = {k: float(np.nanmean([m[k] for m in vals['xgb']])) for k in vals['xgb'][0]}
            avg_dw = {k: float(np.nanmean([m[k] for m in vals['dwrf']])) for k in vals['dwrf'][0]}
            model_results[f"{key}_rf"] = avg_rf
            model_results[f"{key}_dwrf"] = avg_dw
            model_results[f"{key}_xgb"] = avg_xgb

        self.results['models'] = model_results
        logger.info(f"Model result keys: {list(model_results.keys())}")

        # Statistical tests
        base_metric_names = ['MAE', 'RMSE', 'MAPE', 'R2']
        stat_results = {}
        for key, vals in aggregated.items():
            for m in base_metric_names:
                if m not in vals['rf'][0]:
                    continue
                higher_better = m not in ['MAE', 'RMSE', 'MAPE']
                rf_vals = [x[m] for x in vals['rf']]
                dw_vals = [x[m] for x in vals['dwrf']]
                try:
                    _, p = safe_wilcoxon(dw_vals, rf_vals)
                except Exception:
                    p = 1.0
                try:
                    r = compute_rank_biserial_correlation(dw_vals, rf_vals, higher_is_better=higher_better)
                except Exception:
                    r = 0.0
                stat_results[f"{key}_bagging_{m}"] = {
                    'p_value': float(p), 'effect_size': float(r),
                    'comparison': 'dw_vs_rf',
                }
                xgb_vals = [x[m] for x in vals['xgb']]
                try:
                    _, p2 = safe_wilcoxon(dw_vals, xgb_vals)
                except Exception:
                    p2 = 1.0
                try:
                    r2 = compute_rank_biserial_correlation(dw_vals, xgb_vals, higher_is_better=higher_better)
                except Exception:
                    r2 = 0.0
                stat_results[f"{key}_boosting_{m}"] = {
                    'p_value': float(p2), 'effect_size': float(r2),
                    'comparison': 'dw_vs_xgb',
                }
        self.results['statistical_tests'] = stat_results

        first_graph = list(dpw_results.keys())[0]
        primary_family = dw_families[0]
        dpw_for_viz = dpw_results[first_graph]
        viz_alg = primary_family

        self.data_loader.data['y_test'] = _inverse_y(y_test)
        self.data_loader.data['y_pred_rf'] = self.results['predictions'].get('RF')
        self.data_loader.data['y_pred_xgb'] = self.results['predictions'].get('XGBoost')
        self.data_loader.data['y_pred_dwrf'] = self.results['predictions'].get('DARF')

        rf_key = f"{first_graph}_{primary_family}_rf"
        dwrf_key = f"{first_graph}_{primary_family}_dwrf"
        xgb_key = f"{first_graph}_{primary_family}_xgb"
        if rf_key not in model_results:
            for k in model_results:
                if k.endswith('_rf'):
                    rf_key = k
                    break
        if dwrf_key not in model_results:
            for k in model_results:
                if k.endswith('_dwrf'):
                    dwrf_key = k
                    break
        if xgb_key not in model_results:
            for k in model_results:
                if k.endswith('_xgb'):
                    xgb_key = k
                    break

        self.data_loader.data['rf_metrics'] = model_results.get(rf_key, {})
        self.data_loader.data['dwrf_metrics'] = model_results.get(dwrf_key, {})
        self.data_loader.data['xgb_metrics'] = model_results.get(xgb_key, {})
        self.data_loader.data['mape_threshold'] = mape_threshold

        # Feature frequencies
        if True:
            model_rf_base = ModelFactory.create_model(
                self.cfg, 'bagging',
                feature_weights=None,
                num_classes=num_classes,
                feature_names=feature_names
            )
            model_rf_base.fit(X_train, y_train)
            if hasattr(model_rf_base, 'get_feature_selection_frequencies'):
                self.results['feature_frequencies']['RF_baseline'] = \
                    model_rf_base.get_feature_selection_frequencies(feature_names)

            for graph_name, dpw in dpw_results.items():
                fam = dw_families[0] if dw_families else 'bagging'
                model = ModelFactory.create_model(
                    self.cfg, fam,
                    feature_weights=dpw,
                    num_classes=num_classes,
                    feature_names=feature_names
                )
                model.fit(X_train, y_train)
                if hasattr(model, 'get_feature_selection_frequencies'):
                    freq = model.get_feature_selection_frequencies(feature_names)
                    self.results['feature_frequencies'][graph_name] = freq

        # 5. Sensitivity
        if self.cfg.ridge.lambda_sensitivity.get('enabled', False) or \
           self.cfg.sampling.alpha_sensitivity.get('enabled', False):
            data_dict = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'dpw_results': dpw_results,
                'feature_names': feature_names,
                'problem_type': self.cfg.problem.type,
                'n_runs': n_runs,
                'alg_types': alg_types,
                'dpw': dpw_for_viz,
                'alg': viz_alg
            }
            self.run_sensitivity_analysis(data_dict)

        # 6. Temporal
        if self.cfg.evaluation.temporal.enabled and 'Hour' in full_df.columns:
            graph_name = first_graph
            dpw = dpw_for_viz
            dw_family = dw_families[0] if dw_families else 'bagging'

            model_rf_temp = ModelFactory.create_model(
                self.cfg, 'bagging',
                feature_weights=None,
                num_classes=num_classes,
                feature_names=feature_names,
            )
            model_rf_temp.fit(X_train, y_train)
            y_pred_rf_temp = model_rf_temp.predict(X_test)

            model_xgb_temp = ModelFactory.create_model(
                self.cfg, 'boosting',
                feature_weights=None,
                num_classes=num_classes,
                feature_names=feature_names,
            )
            model_xgb_temp.fit(X_train, y_train)
            y_pred_xgb_temp = model_xgb_temp.predict(X_test)

            model_dwrf_temp = ModelFactory.create_model(
                self.cfg, dw_family,
                feature_weights=dpw,
                num_classes=num_classes,
                feature_names=feature_names,
            )
            model_dwrf_temp.fit(X_train, y_train)
            y_pred_dwrf_temp = model_dwrf_temp.predict(X_test)

            if hasattr(self.data_loader, 'test_indices'):
                hours_test = full_df.iloc[self.data_loader.test_indices]['Hour'].values
            else:
                hours_test = full_df['Hour'].values[-len(X_test):]

            y_true_t = _inverse_y(y_test)
            y_rf_t = _inverse_y(y_pred_rf_temp)
            y_xgb_t = _inverse_y(y_pred_xgb_temp)
            y_dw_t = _inverse_y(y_pred_dwrf_temp)

            self.results['temporal'] = self.temporal_analyzer.analyze_time_of_day(
                y_true_t, y_rf_t, y_dw_t, hours_test, y_pred_xgb=y_xgb_t
            )
            hourly = self.temporal_analyzer.analyze_hour_by_hour(
                y_true_t, y_rf_t, y_dw_t, hours_test, y_pred_xgb=y_xgb_t
            )
            self.results['temporal']['hourly'] = hourly
            logger.info("Temporal analysis completed (RF, XGBoost, experimental).")

        # 6b. Reference timing
        try:
            reference_timing = self._measure_reference_timing(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                feature_names=feature_names,
                dpw=dpw_for_viz,
                dw_family=primary_family,
                num_classes=num_classes,
                n_repeats=5,
            )
            reference_timing['Preprocessing'] = preprocessing_time
            self.results['timing'] = reference_timing
            logger.info("Reference timing installed into results['timing'].")
        except Exception as e:
            logger.warning(
                f"Reference timing measurement failed ({e}); "
                "keeping provisional timing from metric runs."
            )

        # 7. Visualizations
        if dpw_raw_results:
            self.generate_visualizations(feature_names, dpw_raw_results, drw_results, self.cfg.graphs)
        else:
            logger.warning("No raw DPW results, skipping visualizations.")

        # 8. Export
        if self.cfg.export.enabled and 'bagging' in alg_types:
            exporter = ONNXExporter(self.cfg)
            model_export = ModelFactory.create_model(
                self.cfg, 'bagging',
                feature_weights=dpw_for_viz,
                num_classes=num_classes,
                feature_names=feature_names
            )
            model_export.fit(X_train, y_train)
            exporter.export(model_export, X_train[:10], feature_names)

        # 9. Save
        self.save_results()
        logger.info("Experiment completed successfully.")
        return self.results

    def run_sensitivity_analysis(self, data_dict):
        if self.cfg.ridge.lambda_sensitivity.get('enabled', False):
            self.results['sensitivity']['lambda'] = self.sensitivity_analyzer.run_lambda_sensitivity(data_dict)
        if self.cfg.sampling.alpha_sensitivity.get('enabled', False):
            self.results['sensitivity']['alpha'] = self.sensitivity_analyzer.run_alpha_sensitivity(data_dict)

    def generate_visualizations(self, feature_names, dpw_raw_results, drw_results, all_graphs):
        if not self.results['models']:
            logger.warning("No model results to visualize. Skipping plots.")
            return

        first_graph = list(dpw_raw_results.keys())[0]
        dpw = dpw_raw_results[first_graph]
        models = self.results['models']

        def _first_key(suffix):
            preferred = [k for k in models if k.startswith(first_graph + '_') and k.endswith(suffix)]
            if preferred:
                return preferred[0]
            any_k = [k for k in models if k.endswith(suffix)]
            return any_k[0] if any_k else None

        rf_key = _first_key('_rf')
        dwrf_key = _first_key('_dwrf')
        xgb_key = _first_key('_xgb')
        logger.info(f"Viz keys: rf={rf_key}, dwrf={dwrf_key}, xgb={xgb_key}")

        self.visualizer = DWRFVisualizer(
            config=self.cfg,
            feature_names=feature_names,
            dpw=dpw,
            dpw_raw=dpw_raw_results,
            drw=drw_results,
            graph_edges=all_graphs,
            rf_summary=models.get(rf_key, {}) if rf_key else {},
            dwrf_summary=models.get(dwrf_key, {}) if dwrf_key else {},
            xgb_summary=models.get(xgb_key, {}) if xgb_key else {},
            multi_graph_results=models,
            sensitivity_results=self.results.get('sensitivity', {}),
            temporal_results=self.results.get('temporal', {}),
            data=self.data_loader.data,
            stat_results=self.results.get('statistical_tests', {}),
            feature_selection_freq=self.results.get('feature_frequencies', {}),
            timing_data=self.results.get('timing', {})
        )
        self.visualizer.generate_all()
        logger.info("Visualizations generated.")

    def save_results(self):
        output_path = self.output_dir / 'results_summary.json'
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)
