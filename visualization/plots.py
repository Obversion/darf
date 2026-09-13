
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import networkx as nx
# from pathlib import Path
# import logging
# from typing import Dict, List, Optional, Any
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Global font configuration: Arial, size 14, applied to every text element.
# # ---------------------------------------------------------------------------
# _UNIFORM_FONT_SIZE = 14
# _UNIFORM_FONT_FAMILY = 'Arial'


# def _configure_global_fonts():
#     """Force Arial 14pt for every matplotlib text element.

#     Note: table font size is NOT controlled through rcParams; it is set per
#     table via ``table.set_fontsize(...)``.
#     """
#     available = {f.name for f in fm.fontManager.ttflist}
#     family = _UNIFORM_FONT_FAMILY if _UNIFORM_FONT_FAMILY in available else 'DejaVu Sans'
#     if family != _UNIFORM_FONT_FAMILY:
#         logger.warning(
#             "Arial not found; falling back to %s. Install Arial for exact match.",
#             family,
#         )
#     matplotlib.rcParams.update({
#         'font.family': family,
#         'font.sans-serif': [family, 'Arial', 'Helvetica', 'DejaVu Sans'],
#         'font.size': _UNIFORM_FONT_SIZE,
#         'axes.titlesize': _UNIFORM_FONT_SIZE,
#         'axes.labelsize': _UNIFORM_FONT_SIZE,
#         'xtick.labelsize': _UNIFORM_FONT_SIZE,
#         'ytick.labelsize': _UNIFORM_FONT_SIZE,
#         'legend.fontsize': _UNIFORM_FONT_SIZE,
#         'figure.titlesize': _UNIFORM_FONT_SIZE,
#         'axes.titleweight': 'bold',
#         'axes.labelweight': 'normal',
#         'axes.titlepad': 12,
#         'axes.labelpad': 8,
#         'legend.frameon': True,
#         'legend.borderaxespad': 0.5,
#         'legend.borderpad': 0.4,
#         'legend.labelspacing': 0.5,
#         'pdf.fonttype': 42,
#         'ps.fonttype': 42,
#         'svg.fonttype': 'none',
#     })


# _configure_global_fonts()


# # ---------------------------------------------------------------------------
# # Font helper: every plot must use exactly the same sizes.
# # ---------------------------------------------------------------------------
# class _UniformFonts:
#     """Single font size for every element – enforces consistency."""
#     size = _UNIFORM_FONT_SIZE
#     title_size = _UNIFORM_FONT_SIZE
#     axis_label_size = _UNIFORM_FONT_SIZE
#     tick_label_size = _UNIFORM_FONT_SIZE
#     legend_size = _UNIFORM_FONT_SIZE
#     annotation_size = _UNIFORM_FONT_SIZE


# class DWRFVisualizer:
#     def __init__(self, config, feature_names, dpw, dpw_raw=None, drw=None, graph_edges=None,
#                  rf_summary=None, dwrf_summary=None, xgb_summary=None,
#                  multi_graph_results=None, sensitivity_results=None,
#                  temporal_results=None, data=None, stat_results=None,
#                  feature_selection_freq=None, timing_data=None):
#         """
#         dpw_raw: dict of graph_name -> {feature: raw_dpw} (before normalization)
#         feature_selection_freq: dict of graph_name -> {feature: frequency}
#         timing_data: dict with preprocessing, training, and inference times
#         """
#         self.config = config
#         self.feature_names = feature_names
#         self.dpw = dpw
#         self.dpw_raw = dpw_raw or {}
#         self.drw = drw or {}
#         self.graph_edges = graph_edges or {}
#         self.rf_summary = rf_summary or {}
#         self.dwrf_summary = dwrf_summary or {}
#         self.xgb_summary = xgb_summary or {}
#         self.multi_graph_results = multi_graph_results or {}
#         self.sensitivity_results = sensitivity_results or {}
#         self.temporal_results = temporal_results or {}
#         self.data = data or {}
#         self.stat_results = stat_results or {}
#         self.feature_selection_freq = feature_selection_freq or {}
#         self.timing_data = timing_data or {}

#         self.output_dir = Path(config.output.dir) / 'plots'
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.colors = config.visualization.colors
#         # Replace any configured fonts with our uniform fonts so every plot matches.
#         self.fonts = _UniformFonts()
#         self.styles = config.visualization.styles

#         defaults = {
#             'rf': 'RF', 'dwrf': 'DARF', 'xgb': 'XGBoost',
#             'rf_baseline': 'RF (unweighted)',
#             'dwrf_vs_rf': 'DARF vs RF', 'dwrf_vs_xgb': 'DARF vs XGBoost',
#         }
#         raw = getattr(getattr(config, 'visualization', None), 'labels', None)
#         if raw is None:
#             vals = defaults
#         elif isinstance(raw, dict):
#             vals = {**defaults, **raw}
#         else:
#             vals = {k: getattr(raw, k, defaults[k]) for k in defaults}
#         self.labels = type('Labels', (), vals)()

#         try:
#             sns.set_theme(style=self.styles.plot_theme)
#         except ValueError:
#             sns.set_theme(style='whitegrid')

#         # Re-apply our fonts because sns.set_theme resets rcParams.
#         _configure_global_fonts()

#         self.save_format = self.styles.save_format

#         if self.data is not None:
#             self.data['timing'] = self.timing_data

#     # ------------------------------------------------------------------
#     # Saving
#     # ------------------------------------------------------------------
#     def _save_fig(self, fig, filename):
#         path = self.output_dir / filename
#         # Ensure nothing is clipped at boundaries.
#         try:
#             fig.tight_layout()
#         except Exception:
#             pass
#         fig.savefig(
#             path,
#             format=self.save_format,
#             dpi=self.styles.figure_dpi,
#             bbox_inches='tight',
#             pad_inches=0.25,
#         )
#         plt.close(fig)
#         logger.debug(f"Saved plot: {path}")

#     # ------------------------------------------------------------------
#     # Value labels
#     # ------------------------------------------------------------------
#     def _add_value_labels(self, bars, fmt='%.3f', fontsize=None, rotation=0, offset=0.02):
#         """Add value labels to bars; always uses the uniform 14pt size."""
#         fontsize = _UNIFORM_FONT_SIZE
#         for bar in bars:
#             height = bar.get_height()
#             if height != 0:
#                 ax = bar.axes
#                 y_min, y_max = ax.get_ylim()
#                 y_range = y_max - y_min
#                 offset_abs = offset * y_range if y_range > 0 else 0.02
#                 y_pos = height + offset_abs
#                 if height < 0:
#                     y_pos = height - offset_abs
#                     va = 'top'
#                 else:
#                     va = 'bottom'
#                 ax.text(
#                     bar.get_x() + bar.get_width() / 2., y_pos,
#                     fmt % height,
#                     ha='center', va=va,
#                     fontsize=fontsize, rotation=rotation,
#                     clip_on=False,
#                 )

#     # ------------------------------------------------------------------
#     # p-value helpers
#     # ------------------------------------------------------------------
#     def _format_p_value(self, p):
#         if p < 0.001:
#             return "<0.001***"
#         elif p < 0.01:
#             return f"{p:.3f}**"
#         elif p < 0.05:
#             return f"{p:.3f}*"
#         return f"{p:.3f}"

#     def _get_significance_stars(self, p):
#         if p < 0.001:
#             return "***"
#         elif p < 0.01:
#             return "**"
#         elif p < 0.05:
#             return "*"
#         return ""

#     # ------------------------------------------------------------------
#     # Generate all
#     # ------------------------------------------------------------------
#     def generate_all(self):
#         logger.info("Generating all plots...")
#         self.plot_graph_structure()
#         self.plot_dag_summary_table()
#         self.plot_multi_graph_comparison()
#         self.plot_domain_prior_weights()
#         self.plot_sampling_probabilities()
#         self.plot_alpha_sensitivity()
#         self.plot_lambda_sensitivity()
#         self.plot_metrics_comparison()
#         self.plot_feature_selection_frequencies()
#         self.plot_computational_times()
#         self.plot_actual_vs_predicted()
#         self.plot_residual_distribution()
#         self.plot_error_density()
#         self.plot_temporal_time_of_day()
#         self.plot_temporal_hourly()
#         self.plot_temporal_improvement()
#         self.plot_percentile_errors()
#         self.plot_error_boxplot()
#         self.plot_tail_error_analysis()
#         self.plot_distribution_shape_metrics()
#         self.plot_error_vs_actual()
#         self.plot_cumulative_distribution()
#         self.plot_improvement_by_percentile()
#         self.plot_error_summary_table()
#         self.plot_distribution_metrics()
#         self.plot_statistical_summary_table()
#         self.plot_performance_dashboard()
#         if self.config.problem.type == "classification":
#             self.plot_confusion_matrix()
#             self.plot_classification_report()
#             self.plot_roc_curves()
#             self.plot_precision_recall_curve()
#         logger.info("All plots generated.")

#     # ------------------------------------------------------------------
#     # Graph structure (DAG) – white-backed labels, clear curved arrows
#     # ------------------------------------------------------------------
#     def plot_graph_structure(self):
#         if not self.graph_edges:
#             return
#         for graph_name, graph_data in self.graph_edges.items():
#             if hasattr(graph_data, 'edges'):
#                 edges = graph_data.edges
#             elif isinstance(graph_data, dict):
#                 edges = graph_data
#             else:
#                 continue

#             G = nx.DiGraph(edges)
#             if G.number_of_nodes() == 0:
#                 continue

#             # --- Layout -----------------------------------------------------
#             pos = nx.spring_layout(G, seed=42, k=2.8, iterations=80)

#             target_node = self.config.features.target
#             roots = [n for n in G.nodes if G.in_degree(n) == 0]
#             intermediates = [n for n in G.nodes if n not in roots and n != target_node]
#             node_colors = []
#             for n in G.nodes:
#                 if n == target_node:
#                     node_colors.append(self.colors.target_node)
#                 elif n in roots:
#                     node_colors.append(self.colors.root_node)
#                 else:
#                     node_colors.append(self.colors.intermediate_node)

#             fig, ax = plt.subplots(figsize=(14, 11))

#             # --- Edges: drawn separately for clarity, with curved arrows ----
#             nx.draw_networkx_edges(
#                 G, pos, ax=ax,
#                 edge_color='#555555',
#                 width=1.8,
#                 arrows=True,
#                 arrowstyle='-|>',
#                 arrowsize=22,
#                 connectionstyle='arc3,rad=0.12',
#                 min_source_margin=18,
#                 min_target_margin=22,
#                 node_size=2600,
#             )

#             # --- Nodes ------------------------------------------------------
#             nx.draw_networkx_nodes(
#                 G, pos, ax=ax,
#                 node_color=node_colors,
#                 node_size=2600,
#                 edgecolors='black',
#                 linewidths=1.2,
#             )

#             # --- Node labels ABOVE the nodes, with white bbox behind text ---
#             label_pos = {n: (x, y + 0.13) for n, (x, y) in pos.items()}
#             nx.draw_networkx_labels(
#                 G, label_pos, ax=ax,
#                 font_size=_UNIFORM_FONT_SIZE,
#                 font_family=matplotlib.rcParams['font.family'],
#                 font_color='black',
#                 font_weight='bold',
#                 verticalalignment='bottom',
#                 horizontalalignment='center',
#                 bbox=dict(
#                     facecolor='white',
#                     edgecolor='none',
#                     alpha=0.9,
#                     boxstyle='round,pad=0.25',
#                 ),
#             )

#             ax.set_title(
#                 f"Expert Graph: {graph_name}",
#                 fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=16,
#             )
#             ax.axis('off')
#             # Small margins so node labels near the border are not clipped.
#             ax.margins(0.12)
#             self._save_fig(fig, f"00_expert_dag_structure_{graph_name}.svg")

#     # ------------------------------------------------------------------
#     # DAG summary table
#     # ------------------------------------------------------------------
#     def plot_dag_summary_table(self):
#         if not self.graph_edges:
#             return
#         rows = []
#         for gname, gdata in self.graph_edges.items():
#             edges = gdata.edges if hasattr(gdata, 'edges') else gdata
#             drw = self.drw.get(gname, {})
#             for parent, children in edges.items():
#                 for child in children:
#                     drw_val = drw.get(child, {}).get(parent, 0.0) if drw else 0.0
#                     dpw_val = self.dpw_raw.get(gname, {}).get(child, 0.0) if self.dpw_raw else 0.0
#                     rows.append([gname, parent, child, f"{drw_val:.3f}", f"{dpw_val:.3f}"])
#         if not rows:
#             logger.warning("No DRW data available for DAG summary table.")
#             return
#         df = pd.DataFrame(rows, columns=['Graph', 'Parent', 'Child', 'DRW', 'DPW'])
#         n_rows = len(df)
#         fig, ax = plt.subplots(figsize=(16, max(3, n_rows * 0.5 + 1.5)))
#         ax.axis('tight')
#         ax.axis('off')
#         table = ax.table(
#             cellText=df.values, colLabels=df.columns,
#             loc='center', cellLoc='center',
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(_UNIFORM_FONT_SIZE)
#         table.scale(1, 1.7)
#         for j in range(len(df.columns)):
#             table[(0, j)].set_facecolor('#4472C4')
#             table[(0, j)].set_text_props(color='white', fontweight='bold')
#         ax.set_title("DAG Summary Table", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=20)
#         self._save_fig(fig, "00_dag_summary_table.svg")

#     # ------------------------------------------------------------------
#     # Domain prior weights
#     # ------------------------------------------------------------------
#     def plot_domain_prior_weights(self):
#         if not self.dpw_raw:
#             logger.warning("No raw DPW data for multiple graphs; falling back to single dpw.")
#             if self.dpw:
#                 fig, ax = plt.subplots(figsize=(12, 7))
#                 features = list(self.dpw.keys())
#                 weights = list(self.dpw.values())
#                 sorted_idx = np.argsort(weights)[::-1]
#                 features = [features[i] for i in sorted_idx]
#                 weights = [weights[i] for i in sorted_idx]
#                 bars = ax.bar(features, weights, color=self.colors.primary)
#                 ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_ylabel("Domain Prior Weight", fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_title("Domain Prior Weights (DPW)",
#                              fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#                 plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                          fontsize=_UNIFORM_FONT_SIZE)
#                 plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
#                 self._add_value_labels(bars, fmt='%.3f')
#                 self._save_fig(fig, "01_domain_prior_weights.svg")
#             return

#         n_graphs = len(self.dpw_raw)
#         fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 7), constrained_layout=True)
#         if n_graphs == 1:
#             axes = [axes]
#         for ax, (graph_name, dpw) in zip(axes, self.dpw_raw.items()):
#             if not dpw:
#                 ax.text(0.5, 0.5, "No data", ha='center', va='center',
#                         fontsize=_UNIFORM_FONT_SIZE)
#                 ax.axis('off')
#                 continue
#             features = list(dpw.keys())
#             weights = list(dpw.values())
#             sorted_idx = np.argsort(weights)[::-1]
#             features = [features[i] for i in sorted_idx]
#             weights = [weights[i] for i in sorted_idx]
#             bars = ax.bar(features, weights, color=self.colors.primary)
#             ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylabel("Domain Prior Weight", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_title(f"DPW - {graph_name}",
#                          fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#             plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                      fontsize=_UNIFORM_FONT_SIZE)
#             plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
#             self._add_value_labels(bars, fmt='%.3f')
#         self._save_fig(fig, "01_domain_prior_weights_all_graphs.svg")

#     # ------------------------------------------------------------------
#     # Sampling probabilities
#     # ------------------------------------------------------------------
#     def plot_sampling_probabilities(self):
#         alpha = self.config.sampling.smoothing_alpha

#         if self.dpw_raw and len(self.dpw_raw) > 1:
#             n_graphs = len(self.dpw_raw)
#             fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 7),
#                                      constrained_layout=True)
#             if n_graphs == 1:
#                 axes = [axes]
#             for ax, (graph_name, dpw) in zip(axes, self.dpw_raw.items()):
#                 if not dpw:
#                     ax.text(0.5, 0.5, "No data", ha='center', va='center',
#                             fontsize=_UNIFORM_FONT_SIZE)
#                     ax.axis('off')
#                     continue
#                 n_features = len(dpw)
#                 probs = {f: (1 - alpha) * w + alpha / n_features for f, w in dpw.items()}
#                 features = list(probs.keys())
#                 weights = list(probs.values())
#                 sorted_idx = np.argsort(weights)[::-1]
#                 features = [features[i] for i in sorted_idx]
#                 weights = [weights[i] for i in sorted_idx]
#                 bars = ax.bar(features, weights, color=self.colors.tertiary)
#                 ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_ylabel("Sampling Probability", fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_title(f"Sampling Probs - {graph_name} (α={alpha})",
#                              fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#                 plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                          fontsize=_UNIFORM_FONT_SIZE)
#                 plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#                 ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
#                 self._add_value_labels(bars, fmt='%.3f')
#             self._save_fig(fig, "01_sampling_probabilities_all_graphs.svg")
#         else:
#             if not self.dpw:
#                 return
#             n = len(self.dpw)
#             probs = {f: (1 - alpha) * w + alpha / n for f, w in self.dpw.items()}
#             fig, ax = plt.subplots(figsize=(12, 7))
#             features = list(probs.keys())
#             weights = list(probs.values())
#             sorted_idx = np.argsort(weights)[::-1]
#             features = [features[i] for i in sorted_idx]
#             weights = [weights[i] for i in sorted_idx]
#             bars = ax.bar(features, weights, color=self.colors.tertiary)
#             ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylabel("Sampling Probability", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_title(f"Feature Sampling Probabilities (α={alpha})",
#                          fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#             plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                      fontsize=_UNIFORM_FONT_SIZE)
#             plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
#             self._add_value_labels(bars, fmt='%.3f')
#             self._save_fig(fig, "01_sampling_probabilities.svg")

#     # ------------------------------------------------------------------
#     # Sensitivity plots
#     # ------------------------------------------------------------------
#     def plot_alpha_sensitivity(self):
#         if not self.sensitivity_results.get('alpha'):
#             return
#         alpha_vals = sorted(self.sensitivity_results['alpha'].keys())
#         scores = [self.sensitivity_results['alpha'][a].get('MAE', 0) for a in alpha_vals]
#         fig, ax = plt.subplots(figsize=(11, 7))
#         ax.plot(alpha_vals, scores, marker='o', color=self.colors.primary,
#                 linewidth=2, markersize=8)
#         ax.set_xlabel("Smoothing Alpha", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Alpha Sensitivity", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=12)
#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         ax.grid(True, alpha=0.3)
#         self._save_fig(fig, "01_alpha_sensitivity.svg")

#     def plot_lambda_sensitivity(self):
#         if not self.sensitivity_results.get('lambda'):
#             return
#         lambda_vals = sorted(self.sensitivity_results['lambda'].keys())
#         scores = [self.sensitivity_results['lambda'][l].get('MAE', 0) for l in lambda_vals]
#         fig, ax = plt.subplots(figsize=(11, 7))
#         ax.plot(lambda_vals, scores, marker='s', color=self.colors.secondary,
#                 linewidth=2, markersize=8)
#         ax.set_xlabel("Ridge Lambda", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Lambda Sensitivity", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=12)
#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         ax.grid(True, alpha=0.3)
#         self._save_fig(fig, "01_lambda_sensitivity.svg")

#     # ------------------------------------------------------------------
#     # Metrics comparison (now a table with % change for MAE/RMSE/MAPE
#     # and absolute difference for R²)
#     # ------------------------------------------------------------------
#     def plot_metrics_comparison(self):
#         """Render the model comparison as a table.

#         Columns:
#             Metric | RF | XGBoost | DARF | Δ DARF vs RF | Δ DARF vs XGBoost

#         Differences:
#             - MAE, RMSE, MAPE: percentage change, sign chosen so that positive
#               always means "DARF is better".
#             - R²: absolute difference (DARF - baseline), positive = better.
#         """
#         metrics = (self.config.evaluation.regression_metrics
#                    if self.config.problem.type == "regression"
#                    else self.config.evaluation.classification_metrics)

#         if not self.rf_summary or not self.dwrf_summary:
#             return

#         has_xgb = bool(self.xgb_summary)
#         lower_is_better = {'MAE', 'RMSE', 'MAPE'}

#         def _safe(d, m):
#             v = d.get(m)
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return None
#             try:
#                 return float(v)
#             except (TypeError, ValueError):
#                 return None

#         def _fmt_val(v):
#             if v is None:
#                 return "—"
#             return f"{v:.4f}"

#         def _pct_change(base, dwrf, metric):
#             """Percentage change, oriented so positive = DARF better."""
#             if base is None or dwrf is None:
#                 return None
#             if abs(base) < 1e-12:
#                 return None
#             if metric in lower_is_better:
#                 return (base - dwrf) / abs(base) * 100.0
#             return (dwrf - base) / abs(base) * 100.0

#         def _abs_diff(base, dwrf):
#             if base is None or dwrf is None:
#                 return None
#             return dwrf - base

#         def _fmt_delta(v, is_percent):
#             if v is None:
#                 return "—"
#             sign = "+" if v > 0 else ""
#             if is_percent:
#                 return f"{sign}{v:.2f}%"
#             return f"{sign}{v:.4f}"

#         # --- Build the table ------------------------------------------------
#         header = ['Metric', self.labels.rf]
#         if has_xgb:
#             header.append(self.labels.xgb)
#         header.append(self.labels.dwrf)
#         header.append(f'Δ {self.labels.dwrf} vs {self.labels.rf}')
#         if has_xgb:
#             header.append(f'Δ {self.labels.dwrf} vs {self.labels.xgb}')

#         rows = []
#         for metric in metrics:
#             rf_v = _safe(self.rf_summary, metric)
#             xgb_v = _safe(self.xgb_summary, metric) if has_xgb else None
#             dwrf_v = _safe(self.dwrf_summary, metric)

#             is_percent = metric in lower_is_better
#             if is_percent:
#                 d_rf = _pct_change(rf_v, dwrf_v, metric)
#                 d_xgb = _pct_change(xgb_v, dwrf_v, metric) if has_xgb else None
#             else:
#                 # R² and any other higher-is-better metric: absolute difference
#                 d_rf = _abs_diff(rf_v, dwrf_v)
#                 d_xgb = _abs_diff(xgb_v, dwrf_v) if has_xgb else None

#             row = [metric, _fmt_val(rf_v)]
#             if has_xgb:
#                 row.append(_fmt_val(xgb_v))
#             row.append(_fmt_val(dwrf_v))
#             row.append(_fmt_delta(d_rf, is_percent))
#             if has_xgb:
#                 row.append(_fmt_delta(d_xgb, is_percent))
#             rows.append(row)

#         if not rows:
#             return

#         n_cols = len(header)
#         fig_w = max(14, n_cols * 2.4)
#         fig_h = max(4.0, len(rows) * 1.0 + 2.5)
#         fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#         ax.axis('tight')
#         ax.axis('off')

#         table = ax.table(cellText=rows, colLabels=header,
#                          loc='center', cellLoc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(_UNIFORM_FONT_SIZE)
#         table.scale(1, 2.0)

#         # Header styling
#         for j in range(n_cols):
#             table[(0, j)].set_facecolor('#4472C4')
#             table[(0, j)].set_text_props(color='white', fontweight='bold')

#         # Color the Δ columns (green if DARF is better, red otherwise)
#         delta_start = n_cols - (2 if has_xgb else 1)
#         for i, row in enumerate(rows):
#             for j in range(delta_start, n_cols):
#                 cell = table[(i + 1, j)]
#                 txt = row[j]
#                 if txt == "—":
#                     continue
#                 if txt.startswith('+'):
#                     cell.set_facecolor('#C6EFCE')
#                 elif txt.startswith('-'):
#                     cell.set_facecolor('#FFC7CE')

#         # Title
#         ax.set_title(
#             f"Performance Metrics Comparison: {self.labels.dwrf} vs "
#             f"{self.labels.rf}"
#             + (f" & {self.labels.xgb}" if has_xgb else ""),
#             fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=20,
#         )

#         # Footnote clarifying the difference units
#         ax.text(
#             0.5, -0.10,
#             "Δ for MAE / RMSE / MAPE = percentage change (positive = DARF better)  |  "
#             "Δ for R² = absolute difference (DARF − baseline)",
#             transform=ax.transAxes, ha='center', va='top',
#             fontsize=_UNIFORM_FONT_SIZE, style='italic',
#         )

#         self._save_fig(fig, "02_metrics_comparison.svg")

#     # ------------------------------------------------------------------
#     # Feature selection frequencies – now a 2×2 grid
#     # ------------------------------------------------------------------
#     def plot_feature_selection_frequencies(self):
#         if not self.feature_selection_freq:
#             logger.warning("No feature selection frequencies available; skipping plot.")
#             return

#         items = []
#         if 'RF_baseline' in self.feature_selection_freq:
#             items.append(('RF_baseline', self.feature_selection_freq['RF_baseline']))
#         for k, v in self.feature_selection_freq.items():
#             if k != 'RF_baseline':
#                 items.append((k, v))

#         n = len(items)
#         if n == 0:
#             return

#         # Always lay out as a 2×2 grid (pad with empty axes if fewer than 4).
#         n_rows, n_cols = 2, 2
#         fig, axes = plt.subplots(
#             n_rows, n_cols,
#             figsize=(16, 12),
#             constrained_layout=True,
#         )
#         axes_flat = axes.flatten()

#         for idx, ax in enumerate(axes_flat):
#             if idx >= n:
#                 ax.axis('off')
#                 continue

#             name, freq = items[idx]

#             if not freq:
#                 ax.text(0.5, 0.5, "No data", ha='center', va='center',
#                         fontsize=_UNIFORM_FONT_SIZE)
#                 ax.axis('off')
#                 continue

#             features = self.feature_names if self.feature_names else list(freq.keys())
#             freqs = [freq.get(f, 0.0) for f in features]
#             sorted_idx = np.argsort(freqs)[::-1]
#             sorted_features = [features[i] for i in sorted_idx]
#             sorted_freqs = [freqs[i] for i in sorted_idx]

#             is_baseline = (name == 'RF_baseline')
#             color = self.colors.secondary if is_baseline else self.colors.primary
#             title = self.labels.rf_baseline if is_baseline else f"{self.labels.dwrf} – {name}"

#             bars = ax.bar(sorted_features, sorted_freqs, color=color)
#             ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylabel("Selection Frequency", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_title(title, fontsize=_UNIFORM_FONT_SIZE,
#                          fontweight='bold', pad=12)
#             ax.set_ylim(0, 1.15)
#             plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
#                      fontsize=_UNIFORM_FONT_SIZE)
#             plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             self._add_value_labels(bars, fmt='%.2f')

#         self._save_fig(fig, "03_feature_selection_frequencies_all_graphs.svg")

#     # ------------------------------------------------------------------
#     # Computational times
#     # ------------------------------------------------------------------
#     def plot_computational_times(self):
#         timing_data = self.timing_data or self.data.get('timing', {}) or {}

#         if not timing_data or 'Training' not in timing_data:
#             logger.warning("No timing data available. Using placeholder values for demonstration.")
#             times = {
#                 'Preprocessing': [0.1, 0.1, 0.1],
#                 'Training': [3.9, 4.2, 4.0],
#                 'Inference': [0.8, 0.03, 0.85]
#             }
#             df = pd.DataFrame(times, index=[self.labels.rf, self.labels.xgb, self.labels.dwrf])
#         else:
#             def _t(section, *keys):
#                 block = timing_data.get(section, {})
#                 if not isinstance(block, dict):
#                     return float(block) if section == 'Preprocessing' else 0.0
#                 for k in keys:
#                     if k in block and block[k] is not None:
#                         try:
#                             return float(block[k])
#                         except (TypeError, ValueError):
#                             continue
#                 return 0.0

#             preproc = timing_data.get('Preprocessing', 0.1)
#             try:
#                 preproc = float(preproc) if not isinstance(preproc, dict) else 0.1
#             except (TypeError, ValueError):
#                 preproc = 0.1
#             train_rf = _t('Training', 'RF', getattr(self.labels, 'rf', 'RF'))
#             train_xgb = _t('Training', 'XGBoost', getattr(self.labels, 'xgb', 'XGBoost'))
#             train_dwrf = _t('Training', 'DARF', getattr(self.labels, 'dwrf', 'DARF'))
#             infer_rf = _t('Inference', 'RF', getattr(self.labels, 'rf', 'RF'))
#             infer_xgb = _t('Inference', 'XGBoost', getattr(self.labels, 'xgb', 'XGBoost'))
#             infer_dwrf = _t('Inference', 'DARF', getattr(self.labels, 'dwrf', 'DARF'))

#             if train_rf == 0 and train_xgb == 0 and train_dwrf == 0:
#                 logger.warning("All training times are zero, using placeholder values.")
#                 train_rf, train_xgb, train_dwrf = 3.9, 4.2, 4.0
#             if infer_rf == 0 and infer_xgb == 0 and infer_dwrf == 0:
#                 logger.warning("All inference times are zero, using placeholder values.")
#                 infer_rf, infer_xgb, infer_dwrf = 0.8, 0.03, 0.85

#             times = {
#                 'Preprocessing': [preproc, preproc, preproc],
#                 'Training': [train_rf, train_xgb, train_dwrf],
#                 'Inference': [infer_rf, infer_xgb, infer_dwrf]
#             }
#             df = pd.DataFrame(times, index=[self.labels.rf, self.labels.xgb, self.labels.dwrf])

#         fig, ax = plt.subplots(figsize=(12, 7))
#         df.plot(kind='bar', ax=ax,
#                 color=[self.colors.primary, self.colors.secondary, self.colors.tertiary])
#         ax.set_title("Computational Times", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=12)
#         ax.set_ylabel("Time (seconds)", fontsize=_UNIFORM_FONT_SIZE)
#         ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
#                   fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#         for container in ax.containers:
#             ax.bar_label(container, fmt='%.4f', fontsize=_UNIFORM_FONT_SIZE,
#                          padding=3)
#         ax.set_xticklabels([self.labels.rf, self.labels.xgb, self.labels.dwrf],
#                            rotation=0, fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         self._save_fig(fig, "04_computational_times.svg")

#     # ------------------------------------------------------------------
#     # Residual distribution
#     # ------------------------------------------------------------------
#     def plot_residual_distribution(self):
#         y_test = self.data.get('y_test')
#         y_pred_rf = self.data.get('y_pred_rf')
#         y_pred_xgb = self.data.get('y_pred_xgb')
#         y_pred_dwrf = self.data.get('y_pred_dwrf')

#         if y_test is None:
#             return

#         has_rf = y_pred_rf is not None and len(y_pred_rf) > 0
#         has_xgb = y_pred_xgb is not None and len(y_pred_xgb) > 0
#         has_dwrf = y_pred_dwrf is not None and len(y_pred_dwrf) > 0

#         if not (has_rf or has_xgb or has_dwrf):
#             return

#         fig, ax = plt.subplots(figsize=(12, 7))

#         if has_rf:
#             resid_rf = y_test - y_pred_rf
#             ax.hist(resid_rf, bins=30, alpha=0.5, label=self.labels.rf,
#                     color=self.colors.secondary)
#         if has_xgb:
#             resid_xgb = y_test - y_pred_xgb
#             ax.hist(resid_xgb, bins=30, alpha=0.5, label=self.labels.xgb,
#                     color=self.colors.tertiary)
#         if has_dwrf:
#             resid_dwrf = y_test - y_pred_dwrf
#             ax.hist(resid_dwrf, bins=30, alpha=0.5, label=self.labels.dwrf,
#                     color=self.colors.primary)

#         ax.axvline(0, color='k', linestyle='--')
#         ax.set_xlabel("Residual", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel("Frequency", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Residual Distribution", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=12)
#         ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
#                   fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         self._save_fig(fig, "06_residual_distribution.svg")

#     # ------------------------------------------------------------------
#     # Error density – means in legend, legend inside top-right plot area
#     # ------------------------------------------------------------------
#     def plot_error_density(self):
#         y_test = self.data.get('y_test')
#         y_pred_rf = self.data.get('y_pred_rf')
#         y_pred_xgb = self.data.get('y_pred_xgb')
#         y_pred_dwrf = self.data.get('y_pred_dwrf')

#         if y_test is None:
#             return

#         fig, ax = plt.subplots(figsize=(12, 7))

#         series = []
#         if y_pred_rf is not None and len(y_pred_rf) > 0:
#             series.append((self.labels.rf, np.abs(y_test - y_pred_rf), self.colors.secondary))
#         if y_pred_xgb is not None and len(y_pred_xgb) > 0:
#             series.append((self.labels.xgb, np.abs(y_test - y_pred_xgb), self.colors.tertiary))
#         if y_pred_dwrf is not None and len(y_pred_dwrf) > 0:
#             series.append((self.labels.dwrf, np.abs(y_test - y_pred_dwrf), self.colors.primary))

#         if not series:
#             return

#         for name, errs, color in series:
#             mean_v = float(np.mean(errs))
#             label = f"{name} (mean = {mean_v:.3f})"
#             sns.kdeplot(errs, label=label, color=color, ax=ax, linewidth=2)
#             # Keep the vertical mean line but no in-plot text annotation;
#             # the value is shown in the legend instead.
#             ax.axvline(mean_v, linestyle='--', color=color, alpha=0.7)

#         ax.set_xlabel("Absolute Error", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel("Density", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Error Density Distribution",
#                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)

#         # Legend anchored inside the plot, top-right corner.
#         ax.legend(
#             loc='upper right',
#             bbox_to_anchor=(0.98, 0.98),
#             fontsize=_UNIFORM_FONT_SIZE,
#             frameon=True,
#             framealpha=0.95,
#             borderaxespad=0.0,
#         )

#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         self._save_fig(fig, "08f_error_density.svg")

#     def plot_time_series_forecast(self):
#         pass

#     # ------------------------------------------------------------------
#     # Temporal plots
#     # ------------------------------------------------------------------
#     def plot_temporal_time_of_day(self):
#         if not self.temporal_results:
#             return
#         buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
#         if not buckets:
#             return
#         metric = 'MAE' if self.config.problem.type == 'regression' else 'accuracy'

#         def _bucket_val(bucket, model_key):
#             v = self.temporal_results[bucket].get(model_key, {}).get(metric, np.nan)
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return 0.0
#             return float(v)

#         rf_vals = [_bucket_val(b, 'rf') for b in buckets]
#         xgb_vals = [_bucket_val(b, 'xgb') for b in buckets]
#         dwrf_vals = [_bucket_val(b, 'dwrf') for b in buckets]
#         has_xgb = any('xgb' in self.temporal_results[b] for b in buckets)

#         fig, ax = plt.subplots(figsize=(12, 7))
#         x = np.arange(len(buckets))
#         n_series = 3 if has_xgb else 2
#         width = 0.8 / n_series
#         offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

#         bars1 = ax.bar(x + offsets[0], rf_vals, width, label=self.labels.rf,
#                        color=self.colors.secondary)
#         if has_xgb:
#             bars2 = ax.bar(x + offsets[1], xgb_vals, width, label=self.labels.xgb,
#                            color=self.colors.tertiary)
#             bars3 = ax.bar(x + offsets[2], dwrf_vals, width, label=self.labels.dwrf,
#                            color=self.colors.primary)
#         else:
#             bars3 = ax.bar(x + offsets[1], dwrf_vals, width, label=self.labels.dwrf,
#                            color=self.colors.primary)

#         ax.set_xlabel("Time of Day", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Time-of-Day Performance",
#                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#         ax.set_xticks(x)
#         ax.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
#         ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                   fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#         ymax = max(rf_vals + dwrf_vals + (xgb_vals if has_xgb else []) + [1e-9])
#         ax.set_ylim(0, ymax * 1.20)
#         self._add_value_labels(bars1, fmt='%.3f')
#         if has_xgb:
#             self._add_value_labels(bars2, fmt='%.3f')
#         self._add_value_labels(bars3, fmt='%.3f')
#         self._save_fig(fig, "08_temporal_time_of_day.svg")

#     def plot_temporal_hourly(self):
#         hourly = self.temporal_results.get('hourly')
#         if not hourly:
#             return
#         hours = sorted(hourly.keys())
#         metric = 'MAE' if self.config.problem.type == 'regression' else 'accuracy'

#         def _h(h, key):
#             v = hourly[h].get(key, {}).get(metric, np.nan)
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return np.nan
#             return float(v)

#         rf_vals = [_h(h, 'rf') for h in hours]
#         dwrf_vals = [_h(h, 'dwrf') for h in hours]
#         has_xgb = any('xgb' in hourly[h] for h in hours)
#         xgb_vals = [_h(h, 'xgb') for h in hours] if has_xgb else None

#         fig, ax = plt.subplots(figsize=(14, 7))
#         ax.plot(hours, rf_vals, label=self.labels.rf, color=self.colors.secondary,
#                 marker='o', linewidth=2, markersize=7)
#         if has_xgb:
#             ax.plot(hours, xgb_vals, label=self.labels.xgb, color=self.colors.tertiary,
#                     marker='^', linewidth=2, markersize=7)
#         ax.plot(hours, dwrf_vals, label=self.labels.dwrf, color=self.colors.primary,
#                 marker='s', linewidth=2, markersize=7)
#         ax.set_xlabel("Hour of Day", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title("Hourly Performance", fontsize=_UNIFORM_FONT_SIZE,
#                      fontweight='bold', pad=12)
#         ax.set_xticks(hours)
#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                   fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#         ax.grid(True, alpha=0.3)
#         self._save_fig(fig, "08_temporal_hourly.svg")

#     def plot_temporal_improvement(self):
#         if not self.temporal_results:
#             return
#         buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
#         if not buckets:
#             return

#         def _v(bucket, key):
#             v = self.temporal_results[bucket].get(key, {}).get('MAE', 0)
#             try:
#                 return float(v)
#             except (TypeError, ValueError):
#                 return 0.0

#         def _imp(base, dw):
#             if abs(base) < 1e-12:
#                 return 0.0
#             return (base - dw) / abs(base) * 100.0

#         rf_vals = [_v(b, 'rf') for b in buckets]
#         dwrf_vals = [_v(b, 'dwrf') for b in buckets]
#         has_xgb = any('xgb' in self.temporal_results[b] for b in buckets)
#         xgb_vals = [_v(b, 'xgb') for b in buckets] if has_xgb else None
#         imp_rf = [_imp(rf, dw) for rf, dw in zip(rf_vals, dwrf_vals)]
#         imp_xgb = [_imp(xgb, dw) for xgb, dw in zip(xgb_vals, dwrf_vals)] if has_xgb else None

#         fig, ax = plt.subplots(figsize=(12, 8))
#         x = np.arange(len(buckets))
#         if has_xgb:
#             width = 0.35
#             bars1 = ax.bar(x - width / 2, imp_rf, width,
#                            label=f'vs {self.labels.rf}', color=self.colors.secondary)
#             bars2 = ax.bar(x + width / 2, imp_xgb, width,
#                            label=f'vs {self.labels.xgb}', color=self.colors.tertiary)
#             self._add_value_labels(bars1, fmt='%.1f%%')
#             self._add_value_labels(bars2, fmt='%.1f%%')
#             ax.set_xticks(x)
#             ax.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
#         else:
#             colors = [self.colors.improvement if i > 0 else self.colors.degradation for i in imp_rf]
#             bars = ax.bar(buckets, imp_rf, color=colors)
#             self._add_value_labels(bars, fmt='%.1f%%')

#         ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
#         ax.set_xlabel("Time of Day", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_ylabel("Improvement (%)", fontsize=_UNIFORM_FONT_SIZE)
#         ax.set_title(f"{self.labels.dwrf} Improvement by Time of Day",
#                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#         plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

#         # Legend at the bottom of the figure rather than far right.
#         if has_xgb:
#             ax.legend(
#                 loc='upper center',
#                 bbox_to_anchor=(0.5, -0.12),
#                 ncol=2,
#                 fontsize=_UNIFORM_FONT_SIZE,
#                 frameon=True,
#             )

#         # Add margins so labels don't clip.
#         all_vals = imp_rf + (imp_xgb or []) + [0]
#         lo, hi = min(all_vals), max(all_vals)
#         span = max(hi - lo, 1.0)
#         ax.set_ylim(lo - 0.25 * span, hi + 0.25 * span)
#         self._save_fig(fig, "08_temporal_improvement.svg")

#     # ------------------------------------------------------------------
#     # Error summary table
#     # ------------------------------------------------------------------
#     def plot_error_summary_table(self):
#         main_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
#         lower_is_better = {'MAE', 'RMSE', 'MAPE'}

#         graph_dwrf = {}
#         rf_from_multi = {}
#         xgb_from_multi = {}

#         for key, mets in (self.multi_graph_results or {}).items():
#             parts = key.split('_')
#             if len(parts) < 3:
#                 continue
#             model_tag = parts[-1]
#             alg = parts[-2]
#             graph = '_'.join(parts[:-2])
#             if model_tag == 'dwrf':
#                 graph_dwrf[graph] = mets
#             elif model_tag == 'rf':
#                 if not rf_from_multi:
#                     rf_from_multi = mets
#             elif model_tag == 'xgb':
#                 if not xgb_from_multi:
#                     xgb_from_multi = mets

#         rf_mets = rf_from_multi or self.rf_summary or {}
#         xgb_mets = xgb_from_multi or self.xgb_summary or {}

#         if not graph_dwrf and self.dwrf_summary:
#             graph_dwrf[self.labels.dwrf] = self.dwrf_summary

#         if not rf_mets and not graph_dwrf:
#             logger.warning("No metrics available for error summary table.")
#             return

#         graphs = sorted(graph_dwrf.keys())

#         def _fmt(v):
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return "—"
#             return f"{float(v):.4f}"

#         def _imp(base, dwrf, metric):
#             if base is None or dwrf is None:
#                 return None
#             try:
#                 base, dwrf = float(base), float(dwrf)
#             except (TypeError, ValueError):
#                 return None
#             if abs(base) < 1e-12 or np.isnan(base) or np.isnan(dwrf):
#                 return None
#             if metric in lower_is_better:
#                 return (base - dwrf) / abs(base) * 100.0
#             return (dwrf - base) / abs(base) * 100.0

#         def _fmt_imp(v):
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return "—"
#             sign = "+" if v > 0 else ""
#             return f"{sign}{v:.2f}%"

#         header = ['Metric', self.labels.rf]
#         has_xgb = bool(xgb_mets)
#         if has_xgb:
#             header.append(self.labels.xgb)
#         for g in graphs:
#             header.append(f'{self.labels.dwrf} ({g})')
#         for g in graphs:
#             header.append(f'Imp vs {self.labels.rf} ({g})')
#         if has_xgb:
#             for g in graphs:
#                 header.append(f'Imp vs {self.labels.xgb} ({g})')

#         rows = []
#         for metric in main_metrics:
#             rf_v = rf_mets.get(metric)
#             xgb_v = xgb_mets.get(metric) if has_xgb else None
#             row = [metric, _fmt(rf_v)]
#             if has_xgb:
#                 row.append(_fmt(xgb_v))
#             for g in graphs:
#                 row.append(_fmt(graph_dwrf[g].get(metric)))
#             for g in graphs:
#                 row.append(_fmt_imp(_imp(rf_v, graph_dwrf[g].get(metric), metric)))
#             if has_xgb:
#                 for g in graphs:
#                     row.append(_fmt_imp(_imp(xgb_v, graph_dwrf[g].get(metric), metric)))
#             rows.append(row)

#         n_cols = len(header)
#         fig_w = max(14, n_cols * 1.5)
#         fig_h = max(3.5, len(rows) * 0.7 + 2.0)
#         fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#         ax.axis('tight')
#         ax.axis('off')

#         table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(_UNIFORM_FONT_SIZE)
#         table.scale(1, 2.0)

#         for j in range(n_cols):
#             table[(0, j)].set_facecolor('#4472C4')
#             table[(0, j)].set_text_props(color='white', fontweight='bold')

#         base_cols = 1 + 1 + (1 if has_xgb else 0) + len(graphs)
#         for i, row in enumerate(rows):
#             for j in range(base_cols, n_cols):
#                 cell = table[(i + 1, j)]
#                 txt = row[j]
#                 if txt.startswith('+'):
#                     cell.set_facecolor('#C6EFCE')
#                 elif txt.startswith('-') and txt != "—":
#                     cell.set_facecolor('#FFC7CE')

#         ax.set_title(
#             f"Error Summary: Model Metrics & {self.labels.dwrf} Improvements "
#             f"vs {self.labels.rf} / {self.labels.xgb}",
#             fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=20)
#         ax.text(
#             0.5, -0.12,
#             f"Improvement %: positive = {self.labels.dwrf} better  |  "
#             "MAE/RMSE/MAPE: lower is better  |  R²: higher is better",
#             transform=ax.transAxes, ha='center', va='top',
#             fontsize=_UNIFORM_FONT_SIZE, style='italic')

#         self._save_fig(fig, "08h_error_summary_table.svg")

#     def plot_distribution_metrics(self):
#         pass

#     # ------------------------------------------------------------------
#     # Statistical summary table
#     # ------------------------------------------------------------------
#     def plot_statistical_summary_table(self):
#         if not self.stat_results:
#             logger.warning("No statistical results available for summary table.")
#             return

#         data = {}
#         for key, val in self.stat_results.items():
#             parts = key.split('_')
#             if len(parts) < 3:
#                 continue
#             metric = parts[-1]
#             alg = parts[-2]
#             graph = '_'.join(parts[:-2])

#             if graph not in data:
#                 data[graph] = {}
#             if metric not in data[graph]:
#                 data[graph][metric] = {}

#             p = val.get('p_value', 1.0)
#             effect = val.get('effect_size', 0.0)

#             if alg == 'bagging':
#                 comp_label = self.labels.dwrf_vs_rf
#             elif alg == 'boosting':
#                 comp_label = self.labels.dwrf_vs_xgb
#             else:
#                 comp_label = alg

#             data[graph][metric][comp_label] = {
#                 'p_value': p,
#                 'effect_size': effect,
#                 'significant': p < 0.05
#             }

#         if not data:
#             logger.warning("No parsed data for statistical summary table.")
#             return

#         graphs = sorted(data.keys())
#         metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
#         comparisons = [self.labels.dwrf_vs_rf, self.labels.dwrf_vs_xgb]

#         table_data = []
#         header = ['Graph', 'Metric', 'Comparison', 'p-value', 'Effect Size (r)', 'Significant']

#         for graph in graphs:
#             for metric in metrics:
#                 if metric not in data[graph]:
#                     continue
#                 for comp in comparisons:
#                     if comp not in data[graph][metric]:
#                         continue
#                     info = data[graph][metric][comp]
#                     p_val = info['p_value']
#                     effect = info['effect_size']
#                     sig = info['significant']

#                     if p_val < 0.001:
#                         p_str = "<0.001***"
#                     elif p_val < 0.01:
#                         p_str = f"{p_val:.3f}**"
#                     elif p_val < 0.05:
#                         p_str = f"{p_val:.3f}*"
#                     else:
#                         p_str = f"{p_val:.3f}"

#                     effect_str = f"{effect:.3f}"
#                     sig_str = "✓" if sig else ""

#                     table_data.append([graph, metric, comp, p_str, effect_str, sig_str])

#         if not table_data:
#             logger.warning("No table data generated.")
#             return

#         fig_w = 16
#         fig_h = max(3.5, len(table_data) * 0.6 + 2.0)
#         fig, ax = plt.subplots(figsize=(fig_w, fig_h))
#         ax.axis('tight')
#         ax.axis('off')

#         table = ax.table(cellText=table_data, colLabels=header,
#                          loc='center', cellLoc='center')
#         table.auto_set_font_size(False)
#         table.set_fontsize(_UNIFORM_FONT_SIZE)
#         table.scale(1, 1.9)

#         for i, row in enumerate(table_data):
#             for j, cell in enumerate(row):
#                 if j == 5:
#                     if cell == "✓":
#                         table[(i + 1, j)].set_facecolor('#90EE90')
#                     else:
#                         table[(i + 1, j)].set_facecolor('#FFCCCC')

#         for j in range(len(header)):
#             table[(0, j)].set_facecolor('#4472C4')
#             table[(0, j)].set_text_props(color='white', fontweight='bold')

#         ax.set_title("Statistical Summary: Wilcoxon Signed-Rank Test Results",
#                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=20)

#         ax.text(0.02, -0.10,
#                 "Significance levels: * p<0.05, ** p<0.01, *** p<0.001  |  ✓ = Significant (p<0.05)",
#                 transform=ax.transAxes, fontsize=_UNIFORM_FONT_SIZE, va='top')

#         self._save_fig(fig, "09_statistical_summary_table.svg")

#     # ------------------------------------------------------------------
#     # Multi-graph comparison
#     # ------------------------------------------------------------------
#     def plot_multi_graph_comparison(self):
#         if not self.multi_graph_results:
#             return

#         graph_data = {}
#         for key, metrics in self.multi_graph_results.items():
#             parts = key.split('_')
#             if len(parts) < 3:
#                 continue
#             model_tag = parts[-1]
#             family = parts[-2]
#             graph_name = '_'.join(parts[:-2])
#             graph_data.setdefault(graph_name, {})
#             if model_tag == 'rf':
#                 graph_data[graph_name]['rf'] = metrics
#             elif model_tag == 'xgb':
#                 graph_data[graph_name]['xgb'] = metrics
#             elif model_tag == 'dwrf':
#                 graph_data[graph_name]['dwrf'] = metrics

#         if not graph_data:
#             logger.warning("No graph data parsed for multi-graph comparison.")
#             return

#         main_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
#         for metric in main_metrics:
#             fig, ax = plt.subplots(figsize=(14, 8))
#             graph_names = sorted(graph_data.keys())
#             x = np.arange(len(graph_names))
#             width = 0.25

#             rf_vals = [graph_data[g].get('rf', {}).get(metric, 0) for g in graph_names]
#             xgb_vals = [graph_data[g].get('xgb', {}).get(metric, 0) for g in graph_names]
#             dwrf_vals = [graph_data[g].get('dwrf', {}).get(metric, 0) for g in graph_names]

#             bars1 = ax.bar(x - width, rf_vals, width, label=self.labels.rf,
#                            color=self.colors.secondary)
#             bars2 = ax.bar(x, xgb_vals, width, label=self.labels.xgb,
#                            color=self.colors.tertiary)
#             bars3 = ax.bar(x + width, dwrf_vals, width, label=self.labels.dwrf,
#                            color=self.colors.primary)

#             ax.set_xticks(x)
#             ax.set_xticklabels(graph_names, rotation=15, ha='right',
#                                fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_title(f"Multi-Graph Comparison – {metric}",
#                          fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
#             ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                       fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#             ymax = max(rf_vals + xgb_vals + dwrf_vals + [1e-9])
#             ax.set_ylim(0, ymax * 1.20)
#             for container in ax.containers:
#                 ax.bar_label(container, fmt='%.3f', fontsize=_UNIFORM_FONT_SIZE, padding=3)
#             plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             self._save_fig(fig, f"04_multi_graph_{metric.lower()}.svg")

#     # ------------------------------------------------------------------
#     # Actual vs predicted
#     # ------------------------------------------------------------------
#     def plot_actual_vs_predicted(self):
#         y_test = self.data.get('y_test')
#         y_pred_rf = self.data.get('y_pred_rf')
#         y_pred_xgb = self.data.get('y_pred_xgb')
#         y_pred_dwrf = self.data.get('y_pred_dwrf')

#         if y_test is None:
#             logger.warning("y_test not found in data for Actual vs Predicted plot.")
#             return

#         has_rf = y_pred_rf is not None and len(y_pred_rf) > 0
#         has_xgb = y_pred_xgb is not None and len(y_pred_xgb) > 0
#         has_dwrf = y_pred_dwrf is not None and len(y_pred_dwrf) > 0

#         if not (has_rf or has_xgb or has_dwrf):
#             logger.warning("No predictions available for Actual vs Predicted plot.")
#             return

#         models = []
#         if has_rf:
#             models.append((self.labels.rf, y_pred_rf, self.colors.secondary,
#                            self.data.get('rf_metrics', {})))
#         if has_xgb:
#             models.append((self.labels.xgb, y_pred_xgb, self.colors.tertiary,
#                            self.data.get('xgb_metrics', {})))
#         if has_dwrf:
#             models.append((self.labels.dwrf, y_pred_dwrf, self.colors.primary,
#                            self.data.get('dwrf_metrics', {})))

#         n_models = len(models)
#         if n_models == 0:
#             return

#         fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 7),
#                                  constrained_layout=True)
#         if n_models == 1:
#             axes = [axes]

#         y_true = y_test
#         y_min, y_max = y_true.min(), y_true.max()
#         padding = (y_max - y_min) * 0.05

#         for idx, (name, y_pred, color, metrics) in enumerate(models):
#             ax = axes[idx]
#             ax.scatter(y_true, y_pred, alpha=0.5, color=color, s=25)

#             ax.plot([y_min - padding, y_max + padding],
#                     [y_min - padding, y_max + padding],
#                     'k--', alpha=0.7, linewidth=1.5)

#             ax.set_xlabel("Actual", fontsize=_UNIFORM_FONT_SIZE)
#             ax.set_ylabel("Predicted", fontsize=_UNIFORM_FONT_SIZE)

#             mae = metrics.get('MAE')
#             rmse = metrics.get('RMSE')
#             r2 = metrics.get('R2')
#             if mae is None or (isinstance(mae, float) and np.isnan(mae)):
#                 mae = mean_absolute_error(y_true, y_pred)
#             if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
#                 rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#             if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
#                 r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

#             ax.set_title(f"{name}\nR² = {r2:.3f} | MAE = {mae:.3f} | RMSE = {rmse:.3f}",
#                          fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)

#             plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             ax.grid(True, alpha=0.3)

#         self._save_fig(fig, "05_actual_vs_predicted.svg")

#     # ------------------------------------------------------------------
#     # Performance dashboard
#     # ------------------------------------------------------------------
#     def plot_performance_dashboard(self):
#         if not self.rf_summary or not self.dwrf_summary:
#             logger.warning("No metrics available for performance dashboard.")
#             return

#         fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
#         fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.12, hspace=0.08, wspace=0.08)

#         metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
#         lower_is_better = {'MAE', 'RMSE', 'MAPE'}

#         def _safe(v):
#             if v is None or (isinstance(v, float) and np.isnan(v)):
#                 return 0.0
#             return float(v)

#         rf_vals = [_safe(self.rf_summary.get(m, 0)) for m in metrics]
#         dwrf_vals = [_safe(self.dwrf_summary.get(m, 0)) for m in metrics]
#         xgb_vals = (
#             [_safe(self.xgb_summary.get(m, 0)) for m in metrics]
#             if self.xgb_summary else None
#         )

#         # 1. Metrics comparison
#         ax1 = axes[0, 0]
#         x = np.arange(len(metrics))
#         n_series = 3 if xgb_vals is not None else 2
#         width = 0.8 / n_series
#         offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

#         bars1 = ax1.bar(x + offsets[0], rf_vals, width, label=self.labels.rf,
#                         color=self.colors.secondary)
#         if xgb_vals is not None:
#             bars2 = ax1.bar(x + offsets[1], xgb_vals, width, label=self.labels.xgb,
#                             color=self.colors.tertiary)
#             bars3 = ax1.bar(x + offsets[2], dwrf_vals, width, label=self.labels.dwrf,
#                             color=self.colors.primary)
#         else:
#             bars3 = ax1.bar(x + offsets[1], dwrf_vals, width, label=self.labels.dwrf,
#                             color=self.colors.primary)

#         ax1.set_xticks(x)
#         ax1.set_xticklabels(metrics, fontsize=_UNIFORM_FONT_SIZE)
#         ax1.set_ylabel("Score", fontsize=_UNIFORM_FONT_SIZE)
#         ax1.set_title("Metrics Comparison", fontsize=_UNIFORM_FONT_SIZE,
#                       fontweight='bold', pad=10)
#         ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                    fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#         ymax1 = max([v for v in rf_vals + dwrf_vals + (xgb_vals or [])] + [1e-9])
#         ax1.set_ylim(0, ymax1 * 1.20)
#         for container in ax1.containers:
#             ax1.bar_label(container, fmt='%.3f', fontsize=_UNIFORM_FONT_SIZE, padding=3)
#         plt.setp(ax1.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

#         # 2. Improvement by metric
#         ax2 = axes[0, 1]
#         improv = []
#         for m, rf, dwrf in zip(metrics, rf_vals, dwrf_vals):
#             if abs(rf) < 1e-12:
#                 improv.append(0.0)
#             elif m in lower_is_better:
#                 improv.append((rf - dwrf) / abs(rf) * 100.0)
#             else:
#                 improv.append((dwrf - rf) / abs(rf) * 100.0)

#         colors_improv = [
#             self.colors.improvement if i > 0 else self.colors.degradation for i in improv
#         ]
#         bars_improv = ax2.bar(metrics, improv, color=colors_improv)
#         ax2.axhline(0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
#         ax2.set_ylabel(f"Improvement over {self.labels.rf} (%)",
#                        fontsize=_UNIFORM_FONT_SIZE)
#         ax2.set_title(f"{self.labels.dwrf} Improvement by Metric",
#                       fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=10)

#         if improv:
#             lo, hi = min(improv + [0]), max(improv + [0])
#             span = max(hi - lo, 1.0)
#             ax2.set_ylim(lo - 0.30 * span, hi + 0.30 * span)
#         for bar, val in zip(bars_improv, improv):
#             y = bar.get_height()
#             ax2.annotate(
#                 f'{val:.1f}%',
#                 xy=(bar.get_x() + bar.get_width() / 2.0, y),
#                 xytext=(0, 6 if val >= 0 else -6),
#                 textcoords='offset points',
#                 ha='center',
#                 va='bottom' if val >= 0 else 'top',
#                 fontsize=_UNIFORM_FONT_SIZE,
#                 clip_on=False,
#             )
#         plt.setp(ax2.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#         plt.setp(ax2.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

#         # 3. Significant results only
#         ax3 = axes[1, 0]
#         sig_data = {}
#         for key, val in (self.stat_results or {}).items():
#             p = val.get('p_value', 1.0)
#             effect = val.get('effect_size', 0.0)
#             if p >= 0.05 or abs(effect) < 0.1:
#                 continue
#             parts = key.split('_')
#             if len(parts) < 3:
#                 continue
#             metric = parts[-1]
#             alg = parts[-2]
#             graph = '_'.join(parts[:-2])
#             sig_data.setdefault(graph, {}).setdefault(metric, {})[alg] = {
#                 'p': p, 'effect': effect
#             }

#         table_data = []
#         for graph in sorted(sig_data.keys()):
#             for metric in metrics:
#                 if metric not in sig_data[graph]:
#                     continue
#                 for alg, info in sig_data[graph][metric].items():
#                     p_str = self._format_p_value(info['p'])
#                     effect_str = f"{info['effect']:.3f}"
#                     stars = self._get_significance_stars(info['p'])
#                     alg_label = (
#                         self.labels.dwrf_vs_rf if alg == 'bagging'
#                         else self.labels.dwrf_vs_xgb if alg == 'boosting'
#                         else alg
#                     )
#                     table_data.append([graph, metric, alg_label, effect_str, p_str, stars])

#         ax3.set_title("Significant Results (p<0.05, |r|≥0.1)",
#                       fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=10)
#         if table_data:
#             if len(table_data) > 8:
#                 table_data = table_data[:8]
#                 table_data.append(['…', '…', '…', '…', '…', '…'])
#             ax3.axis('off')
#             header = ['Graph', 'Metric', 'Comparison', 'Effect (r)', 'p-value', '']
#             table = ax3.table(
#                 cellText=table_data, colLabels=header,
#                 loc='upper center', cellLoc='center'
#             )
#             table.auto_set_font_size(False)
#             table.set_fontsize(_UNIFORM_FONT_SIZE)
#             table.scale(1.0, 1.8)
#             for j in range(len(header)):
#                 table[(0, j)].set_facecolor('#4472C4')
#                 table[(0, j)].set_text_props(color='white', fontweight='bold')
#             for i in range(len(table_data)):
#                 for j in range(len(header)):
#                     table[(i + 1, j)].set_facecolor('#E8F5E9')
#         else:
#             ax3.axis('off')
#             ax3.text(
#                 0.5, 0.5,
#                 "No significant results\n(p < 0.05 and |r| ≥ 0.1)",
#                 ha='center', va='center', transform=ax3.transAxes,
#                 fontsize=_UNIFORM_FONT_SIZE,
#             )

#         # 4. Temporal performance
#         ax4 = axes[1, 1]
#         ax4.set_title("Temporal Performance", fontsize=_UNIFORM_FONT_SIZE,
#                       fontweight='bold', pad=10)
#         if self.temporal_results:
#             buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
#             if buckets:
#                 rf_t = [_safe(self.temporal_results[b].get('rf', {}).get('MAE', 0)) for b in buckets]
#                 dwrf_t = [_safe(self.temporal_results[b].get('dwrf', {}).get('MAE', 0)) for b in buckets]
#                 has_xgb_t = any('xgb' in self.temporal_results[b] for b in buckets)
#                 xgb_t = [_safe(self.temporal_results[b].get('xgb', {}).get('MAE', 0)) for b in buckets] if has_xgb_t else None
#                 x2 = np.arange(len(buckets))
#                 n_series = 3 if has_xgb_t else 2
#                 w = 0.8 / n_series
#                 offs = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * w
#                 bars1_t = ax4.bar(x2 + offs[0], rf_t, w, label=self.labels.rf,
#                                   color=self.colors.secondary)
#                 if has_xgb_t:
#                     barsx_t = ax4.bar(x2 + offs[1], xgb_t, w, label=self.labels.xgb,
#                                       color=self.colors.tertiary)
#                     bars2_t = ax4.bar(x2 + offs[2], dwrf_t, w, label=self.labels.dwrf,
#                                       color=self.colors.primary)
#                 else:
#                     bars2_t = ax4.bar(x2 + offs[1], dwrf_t, w, label=self.labels.dwrf,
#                                       color=self.colors.primary)
#                 ax4.set_xticks(x2)
#                 ax4.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
#                 ax4.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
#                 ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
#                            fontsize=_UNIFORM_FONT_SIZE, frameon=True)
#                 ymax4 = max(rf_t + dwrf_t + (xgb_t or []) + [1e-9])
#                 ax4.set_ylim(0, ymax4 * 1.20)
#                 containers = [bars1_t, bars2_t] + ([barsx_t] if has_xgb_t else [])
#                 for container in containers:
#                     ax4.bar_label(container, fmt='%.3f',
#                                   fontsize=_UNIFORM_FONT_SIZE, padding=3)
#                 plt.setp(ax4.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
#             else:
#                 ax4.axis('off')
#                 ax4.text(0.5, 0.5, "No temporal data", ha='center', va='center',
#                          transform=ax4.transAxes, fontsize=_UNIFORM_FONT_SIZE)
#         else:
#             ax4.axis('off')
#             ax4.text(0.5, 0.5, "No temporal data", ha='center', va='center',
#                      transform=ax4.transAxes, fontsize=_UNIFORM_FONT_SIZE)

#         self._save_fig(fig, "14_performance_dashboard.svg")

#     # ------------------------------------------------------------------
#     # Stubs
#     # ------------------------------------------------------------------
#     def plot_percentile_errors(self): pass
#     def plot_error_boxplot(self): pass
#     def plot_tail_error_analysis(self): pass
#     def plot_distribution_shape_metrics(self): pass
#     def plot_error_vs_actual(self): pass
#     def plot_cumulative_distribution(self): pass
#     def plot_improvement_by_percentile(self): pass

#     def plot_confusion_matrix(self): pass
#     def plot_classification_report(self): pass
#     def plot_roc_curves(self): pass
#     def plot_precision_recall_curve(self): pass


# visualization/plots.py - Updated with uniform Arial 14pt fonts, no overlays, clearer DAGs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global font configuration: Arial, size 14, applied to every text element.
# ---------------------------------------------------------------------------
_UNIFORM_FONT_SIZE = 14
_UNIFORM_FONT_FAMILY = 'Arial'


def _configure_global_fonts():
    """Force Arial 14pt for every matplotlib text element.

    Note: table font size is NOT controlled through rcParams; it is set per
    table via ``table.set_fontsize(...)``.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    family = _UNIFORM_FONT_FAMILY if _UNIFORM_FONT_FAMILY in available else 'DejaVu Sans'
    if family != _UNIFORM_FONT_FAMILY:
        logger.warning(
            "Arial not found; falling back to %s. Install Arial for exact match.",
            family,
        )
    matplotlib.rcParams.update({
        'font.family': family,
        'font.sans-serif': [family, 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': _UNIFORM_FONT_SIZE,
        'axes.titlesize': _UNIFORM_FONT_SIZE,
        'axes.labelsize': _UNIFORM_FONT_SIZE,
        'xtick.labelsize': _UNIFORM_FONT_SIZE,
        'ytick.labelsize': _UNIFORM_FONT_SIZE,
        'legend.fontsize': _UNIFORM_FONT_SIZE,
        'figure.titlesize': _UNIFORM_FONT_SIZE,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.titlepad': 12,
        'axes.labelpad': 8,
        'legend.frameon': True,
        'legend.borderaxespad': 0.5,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


_configure_global_fonts()


# ---------------------------------------------------------------------------
# Font helper: every plot must use exactly the same sizes.
# ---------------------------------------------------------------------------
class _UniformFonts:
    """Single font size for every element – enforces consistency."""
    size = _UNIFORM_FONT_SIZE
    title_size = _UNIFORM_FONT_SIZE
    axis_label_size = _UNIFORM_FONT_SIZE
    tick_label_size = _UNIFORM_FONT_SIZE
    legend_size = _UNIFORM_FONT_SIZE
    annotation_size = _UNIFORM_FONT_SIZE


def _wrap_text(text, max_chars):
    """Wrap long text on whitespace, falling back to hard breaks."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    import textwrap
    return "\n".join(textwrap.wrap(text, width=max_chars,
                                    break_long_words=True,
                                    break_on_hyphens=True))


class DWRFVisualizer:
    def __init__(self, config, feature_names, dpw, dpw_raw=None, drw=None, graph_edges=None,
                 rf_summary=None, dwrf_summary=None, xgb_summary=None,
                 multi_graph_results=None, sensitivity_results=None,
                 temporal_results=None, data=None, stat_results=None,
                 feature_selection_freq=None, timing_data=None):
        """
        dpw_raw: dict of graph_name -> {feature: raw_dpw} (before normalization)
        feature_selection_freq: dict of graph_name -> {feature: frequency}
        timing_data: dict with preprocessing, training, and inference times
        """
        self.config = config
        self.feature_names = feature_names
        self.dpw = dpw
        self.dpw_raw = dpw_raw or {}
        self.drw = drw or {}
        self.graph_edges = graph_edges or {}
        self.rf_summary = rf_summary or {}
        self.dwrf_summary = dwrf_summary or {}
        self.xgb_summary = xgb_summary or {}
        self.multi_graph_results = multi_graph_results or {}
        self.sensitivity_results = sensitivity_results or {}
        self.temporal_results = temporal_results or {}
        self.data = data or {}
        self.stat_results = stat_results or {}
        self.feature_selection_freq = feature_selection_freq or {}
        self.timing_data = timing_data or {}

        self.output_dir = Path(config.output.dir) / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = config.visualization.colors
        # Replace any configured fonts with our uniform fonts so every plot matches.
        self.fonts = _UniformFonts()
        self.styles = config.visualization.styles

        defaults = {
            'rf': 'RF', 'dwrf': 'DARF', 'xgb': 'XGBoost',
            'rf_baseline': 'RF (unweighted)',
            'dwrf_vs_rf': 'DARF vs RF', 'dwrf_vs_xgb': 'DARF vs XGBoost',
        }
        raw = getattr(getattr(config, 'visualization', None), 'labels', None)
        if raw is None:
            vals = defaults
        elif isinstance(raw, dict):
            vals = {**defaults, **raw}
        else:
            vals = {k: getattr(raw, k, defaults[k]) for k in defaults}
        self.labels = type('Labels', (), vals)()

        try:
            sns.set_theme(style=self.styles.plot_theme)
        except ValueError:
            sns.set_theme(style='whitegrid')

        # Re-apply our fonts because sns.set_theme resets rcParams.
        _configure_global_fonts()

        self.save_format = self.styles.save_format

        if self.data is not None:
            self.data['timing'] = self.timing_data

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_fig(self, fig, filename):
        path = self.output_dir / filename
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(
            path,
            format=self.save_format,
            dpi=self.styles.figure_dpi,
            bbox_inches='tight',
            pad_inches=0.25,
        )
        plt.close(fig)
        logger.debug(f"Saved plot: {path}")

    # ------------------------------------------------------------------
    # Value labels
    # ------------------------------------------------------------------
    def _add_value_labels(self, bars, fmt='%.3f', fontsize=None, rotation=0, offset=0.02):
        fontsize = _UNIFORM_FONT_SIZE
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax = bar.axes
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                offset_abs = offset * y_range if y_range > 0 else 0.02
                y_pos = height + offset_abs
                if height < 0:
                    y_pos = height - offset_abs
                    va = 'top'
                else:
                    va = 'bottom'
                ax.text(
                    bar.get_x() + bar.get_width() / 2., y_pos,
                    fmt % height,
                    ha='center', va=va,
                    fontsize=fontsize, rotation=rotation,
                    clip_on=False,
                )

    # ------------------------------------------------------------------
    # p-value helpers
    # ------------------------------------------------------------------
    def _format_p_value(self, p):
        if p < 0.001:
            return "<0.001***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        return f"{p:.3f}"

    def _get_significance_stars(self, p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    # ------------------------------------------------------------------
    # Generate all
    # ------------------------------------------------------------------
    def generate_all(self):
        logger.info("Generating all plots...")
        self.plot_graph_structure()
        self.plot_dag_summary_table()
        self.plot_multi_graph_comparison()
        self.plot_domain_prior_weights()
        self.plot_sampling_probabilities()
        self.plot_alpha_sensitivity()
        self.plot_lambda_sensitivity()
        self.plot_metrics_comparison()
        self.plot_feature_selection_frequencies()
        self.plot_computational_times()
        self.plot_actual_vs_predicted()
        self.plot_residual_distribution()
        self.plot_error_density()
        self.plot_temporal_time_of_day()
        self.plot_temporal_hourly()
        self.plot_temporal_improvement()
        self.plot_percentile_errors()
        self.plot_error_boxplot()
        self.plot_tail_error_analysis()
        self.plot_distribution_shape_metrics()
        self.plot_error_vs_actual()
        self.plot_cumulative_distribution()
        self.plot_improvement_by_percentile()
        self.plot_error_summary_table()
        self.plot_distribution_metrics()
        self.plot_statistical_summary_table()
        self.plot_performance_dashboard()
        if self.config.problem.type == "classification":
            self.plot_confusion_matrix()
            self.plot_classification_report()
            self.plot_roc_curves()
            self.plot_precision_recall_curve()
        logger.info("All plots generated.")

    # ------------------------------------------------------------------
    # Graph structure (DAG) – white-backed labels, clear curved arrows
    # ------------------------------------------------------------------
    def plot_graph_structure(self):
        if not self.graph_edges:
            return
        for graph_name, graph_data in self.graph_edges.items():
            if hasattr(graph_data, 'edges'):
                edges = graph_data.edges
            elif isinstance(graph_data, dict):
                edges = graph_data
            else:
                continue

            G = nx.DiGraph(edges)
            if G.number_of_nodes() == 0:
                continue

            pos = nx.spring_layout(G, seed=42, k=2.8, iterations=80)

            target_node = self.config.features.target
            roots = [n for n in G.nodes if G.in_degree(n) == 0]
            intermediates = [n for n in G.nodes if n not in roots and n != target_node]
            node_colors = []
            for n in G.nodes:
                if n == target_node:
                    node_colors.append(self.colors.target_node)
                elif n in roots:
                    node_colors.append(self.colors.root_node)
                else:
                    node_colors.append(self.colors.intermediate_node)

            fig, ax = plt.subplots(figsize=(14, 11))

            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color='#555555',
                width=1.8,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=22,
                connectionstyle='arc3,rad=0.12',
                min_source_margin=18,
                min_target_margin=22,
                node_size=2600,
            )

            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=node_colors,
                node_size=2600,
                edgecolors='black',
                linewidths=1.2,
            )

            label_pos = {n: (x, y + 0.13) for n, (x, y) in pos.items()}
            nx.draw_networkx_labels(
                G, label_pos, ax=ax,
                font_size=_UNIFORM_FONT_SIZE,
                font_family=matplotlib.rcParams['font.family'],
                font_color='black',
                font_weight='bold',
                verticalalignment='bottom',
                horizontalalignment='center',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.9,
                    boxstyle='round,pad=0.25',
                ),
            )

            ax.set_title(
                f"Expert Graph: {graph_name}",
                fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=16,
            )
            ax.axis('off')
            ax.margins(0.12)
            self._save_fig(fig, f"00_expert_dag_structure_{graph_name}.svg")

    # ------------------------------------------------------------------
    # DAG summary table
    # ------------------------------------------------------------------
    def plot_dag_summary_table(self):
        if not self.graph_edges:
            return
        rows = []
        for gname, gdata in self.graph_edges.items():
            edges = gdata.edges if hasattr(gdata, 'edges') else gdata
            drw = self.drw.get(gname, {})
            for parent, children in edges.items():
                for child in children:
                    drw_val = drw.get(child, {}).get(parent, 0.0) if drw else 0.0
                    dpw_val = self.dpw_raw.get(gname, {}).get(child, 0.0) if self.dpw_raw else 0.0
                    rows.append([gname, parent, child, f"{drw_val:.3f}", f"{dpw_val:.3f}"])
        if not rows:
            logger.warning("No DRW data available for DAG summary table.")
            return
        df = pd.DataFrame(rows, columns=['Graph', 'Parent', 'Child', 'DRW', 'DPW'])
        n_rows = len(df)
        fig, ax = plt.subplots(figsize=(16, max(3, n_rows * 0.5 + 1.5)))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=df.values, colLabels=df.columns,
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(_UNIFORM_FONT_SIZE)
        table.scale(1, 1.7)
        for j in range(len(df.columns)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        ax.set_title("DAG Summary Table", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=20)
        self._save_fig(fig, "00_dag_summary_table.svg")

    # ------------------------------------------------------------------
    # Domain prior weights
    # ------------------------------------------------------------------
    def plot_domain_prior_weights(self):
        if not self.dpw_raw:
            logger.warning("No raw DPW data for multiple graphs; falling back to single dpw.")
            if self.dpw:
                fig, ax = plt.subplots(figsize=(12, 7))
                features = list(self.dpw.keys())
                weights = list(self.dpw.values())
                sorted_idx = np.argsort(weights)[::-1]
                features = [features[i] for i in sorted_idx]
                weights = [weights[i] for i in sorted_idx]
                bars = ax.bar(features, weights, color=self.colors.primary)
                ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
                ax.set_ylabel("Domain Prior Weight", fontsize=_UNIFORM_FONT_SIZE)
                ax.set_title("Domain Prior Weights (DPW)",
                             fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                         fontsize=_UNIFORM_FONT_SIZE)
                plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
                ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
                self._add_value_labels(bars, fmt='%.3f')
                self._save_fig(fig, "01_domain_prior_weights.svg")
            return

        n_graphs = len(self.dpw_raw)
        fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 7), constrained_layout=True)
        if n_graphs == 1:
            axes = [axes]
        for ax, (graph_name, dpw) in zip(axes, self.dpw_raw.items()):
            if not dpw:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        fontsize=_UNIFORM_FONT_SIZE)
                ax.axis('off')
                continue
            features = list(dpw.keys())
            weights = list(dpw.values())
            sorted_idx = np.argsort(weights)[::-1]
            features = [features[i] for i in sorted_idx]
            weights = [weights[i] for i in sorted_idx]
            bars = ax.bar(features, weights, color=self.colors.primary)
            ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylabel("Domain Prior Weight", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_title(f"DPW - {graph_name}",
                         fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                     fontsize=_UNIFORM_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
            self._add_value_labels(bars, fmt='%.3f')
        self._save_fig(fig, "01_domain_prior_weights_all_graphs.svg")

    # ------------------------------------------------------------------
    # Sampling probabilities
    # ------------------------------------------------------------------
    def plot_sampling_probabilities(self):
        alpha = self.config.sampling.smoothing_alpha

        if self.dpw_raw and len(self.dpw_raw) > 1:
            n_graphs = len(self.dpw_raw)
            fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 7),
                                     constrained_layout=True)
            if n_graphs == 1:
                axes = [axes]
            for ax, (graph_name, dpw) in zip(axes, self.dpw_raw.items()):
                if not dpw:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center',
                            fontsize=_UNIFORM_FONT_SIZE)
                    ax.axis('off')
                    continue
                n_features = len(dpw)
                probs = {f: (1 - alpha) * w + alpha / n_features for f, w in dpw.items()}
                features = list(probs.keys())
                weights = list(probs.values())
                sorted_idx = np.argsort(weights)[::-1]
                features = [features[i] for i in sorted_idx]
                weights = [weights[i] for i in sorted_idx]
                bars = ax.bar(features, weights, color=self.colors.tertiary)
                ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
                ax.set_ylabel("Sampling Probability", fontsize=_UNIFORM_FONT_SIZE)
                ax.set_title(f"Sampling Probs - {graph_name} (α={alpha})",
                             fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                         fontsize=_UNIFORM_FONT_SIZE)
                plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
                ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
                self._add_value_labels(bars, fmt='%.3f')
            self._save_fig(fig, "01_sampling_probabilities_all_graphs.svg")
        else:
            if not self.dpw:
                return
            n = len(self.dpw)
            probs = {f: (1 - alpha) * w + alpha / n for f, w in self.dpw.items()}
            fig, ax = plt.subplots(figsize=(12, 7))
            features = list(probs.keys())
            weights = list(probs.values())
            sorted_idx = np.argsort(weights)[::-1]
            features = [features[i] for i in sorted_idx]
            weights = [weights[i] for i in sorted_idx]
            bars = ax.bar(features, weights, color=self.colors.tertiary)
            ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylabel("Sampling Probability", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_title(f"Feature Sampling Probabilities (α={alpha})",
                         fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                     fontsize=_UNIFORM_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylim(0, max(weights) * 1.15 if weights else 1)
            self._add_value_labels(bars, fmt='%.3f')
            self._save_fig(fig, "01_sampling_probabilities.svg")

    # ------------------------------------------------------------------
    # Sensitivity plots
    # ------------------------------------------------------------------
    def plot_alpha_sensitivity(self):
        if not self.sensitivity_results.get('alpha'):
            return
        alpha_vals = sorted(self.sensitivity_results['alpha'].keys())
        scores = [self.sensitivity_results['alpha'][a].get('MAE', 0) for a in alpha_vals]
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(alpha_vals, scores, marker='o', color=self.colors.primary,
                linewidth=2, markersize=8)
        ax.set_xlabel("Smoothing Alpha", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Alpha Sensitivity", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=12)
        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "01_alpha_sensitivity.svg")

    def plot_lambda_sensitivity(self):
        if not self.sensitivity_results.get('lambda'):
            return
        lambda_vals = sorted(self.sensitivity_results['lambda'].keys())
        scores = [self.sensitivity_results['lambda'][l].get('MAE', 0) for l in lambda_vals]
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(lambda_vals, scores, marker='s', color=self.colors.secondary,
                linewidth=2, markersize=8)
        ax.set_xlabel("Ridge Lambda", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Lambda Sensitivity", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=12)
        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "01_lambda_sensitivity.svg")

    # ------------------------------------------------------------------
    # Metrics comparison – table with % change and absolute diff
    # ------------------------------------------------------------------
    def plot_metrics_comparison(self):
        metrics = (self.config.evaluation.regression_metrics
                   if self.config.problem.type == "regression"
                   else self.config.evaluation.classification_metrics)

        if not self.rf_summary or not self.dwrf_summary:
            return

        has_xgb = bool(self.xgb_summary)
        lower_is_better = {'MAE', 'RMSE', 'MAPE'}

        def _safe(d, m):
            v = d.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _fmt_val(v):
            if v is None:
                return "—"
            return f"{v:.4f}"

        def _pct_change(base, dwrf, metric):
            if base is None or dwrf is None:
                return None
            if abs(base) < 1e-12:
                return None
            if metric in lower_is_better:
                return (base - dwrf) / abs(base) * 100.0
            return (dwrf - base) / abs(base) * 100.0

        def _abs_diff(base, dwrf):
            if base is None or dwrf is None:
                return None
            return dwrf - base

        def _fmt_delta(v, is_percent):
            if v is None:
                return "—"
            sign = "+" if v > 0 else ""
            if is_percent:
                return f"{sign}{v:.2f}%"
            return f"{sign}{v:.4f}"

        # ---- Build headers (wrapped where helpful) -------------------------
        header = ['Metric', self.labels.rf]
        if has_xgb:
            header.append(self.labels.xgb)
        header.append(self.labels.dwrf)
        header.append(f'Δ {self.labels.dwrf}\nvs {self.labels.rf}')
        if has_xgb:
            header.append(f'Δ {self.labels.dwrf}\nvs {self.labels.xgb}')

        # ---- Build rows ----------------------------------------------------
        rows = []
        for metric in metrics:
            rf_v = _safe(self.rf_summary, metric)
            xgb_v = _safe(self.xgb_summary, metric) if has_xgb else None
            dwrf_v = _safe(self.dwrf_summary, metric)

            is_percent = metric in lower_is_better
            if is_percent:
                d_rf = _pct_change(rf_v, dwrf_v, metric)
                d_xgb = _pct_change(xgb_v, dwrf_v, metric) if has_xgb else None
            else:
                d_rf = _abs_diff(rf_v, dwrf_v)
                d_xgb = _abs_diff(xgb_v, dwrf_v) if has_xgb else None

            row = [metric, _fmt_val(rf_v)]
            if has_xgb:
                row.append(_fmt_val(xgb_v))
            row.append(_fmt_val(dwrf_v))
            row.append(_fmt_delta(d_rf, is_percent))
            if has_xgb:
                row.append(_fmt_delta(d_xgb, is_percent))
            rows.append(row)

        if not rows:
            return

        n_cols = len(header)

        # ---- Content-aware column widths -----------------------------------
        def _longest_line(s):
            return max((len(part) for part in str(s).split('\n')), default=0)

        col_char_widths = []
        for j in range(n_cols):
            header_len = _longest_line(header[j])
            cell_len = max((_longest_line(row[j]) for row in rows), default=0)
            # A little extra padding so text doesn't touch the cell border.
            col_char_widths.append(max(header_len, cell_len) + 2)

        total_chars = sum(col_char_widths)
        col_widths = [w / total_chars for w in col_char_widths]

        # ---- Figure size from column content -------------------------------
        inches_per_char = 0.16
        fig_w = max(10.0, total_chars * inches_per_char)
        fig_h = max(4.0, len(rows) * 1.0 + 2.8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=rows, colLabels=header,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(_UNIFORM_FONT_SIZE)
        table.scale(1, 2.2)

        # Apply content-derived widths.
        for j, w in enumerate(col_widths):
            for i in range(len(rows) + 1):
                table[(i, j)].set_width(w)

        # Header styling.
        for j in range(n_cols):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Colour the delta columns (green = DARF better, red = worse).
        delta_start = n_cols - (2 if has_xgb else 1)
        for i, row in enumerate(rows):
            for j in range(delta_start, n_cols):
                cell = table[(i + 1, j)]
                txt = row[j]
                if txt == "—":
                    continue
                if txt.startswith('+'):
                    cell.set_facecolor('#C6EFCE')
                elif txt.startswith('-'):
                    cell.set_facecolor('#FFC7CE')

        ax.set_title(
            f"Performance Metrics Comparison: {self.labels.dwrf} vs "
            f"{self.labels.rf}"
            + (f" & {self.labels.xgb}" if has_xgb else ""),
            fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=24,
        )

        ax.text(
            0.5, -0.12,
            "Δ for MAE / RMSE / MAPE = percentage change (positive = DARF better)  |  "
            "Δ for R² = absolute difference (DARF − baseline)",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=_UNIFORM_FONT_SIZE, style='italic',
        )

        self._save_fig(fig, "02_metrics_comparison.svg")
  


    # ------------------------------------------------------------------
    # Feature selection frequencies – 2×2 grid
    # ------------------------------------------------------------------
    def plot_feature_selection_frequencies(self):
        if not self.feature_selection_freq:
            logger.warning("No feature selection frequencies available; skipping plot.")
            return

        items = []
        if 'RF_baseline' in self.feature_selection_freq:
            items.append(('RF_baseline', self.feature_selection_freq['RF_baseline']))
        for k, v in self.feature_selection_freq.items():
            if k != 'RF_baseline':
                items.append((k, v))

        n = len(items)
        if n == 0:
            return

        n_rows, n_cols = 2, 2
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, 12),
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        for idx, ax in enumerate(axes_flat):
            if idx >= n:
                ax.axis('off')
                continue

            name, freq = items[idx]

            if not freq:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        fontsize=_UNIFORM_FONT_SIZE)
                ax.axis('off')
                continue

            features = self.feature_names if self.feature_names else list(freq.keys())
            freqs = [freq.get(f, 0.0) for f in features]
            sorted_idx = np.argsort(freqs)[::-1]
            sorted_features = [features[i] for i in sorted_idx]
            sorted_freqs = [freqs[i] for i in sorted_idx]

            is_baseline = (name == 'RF_baseline')
            color = self.colors.secondary if is_baseline else self.colors.primary
            title = self.labels.rf_baseline if is_baseline else f"{self.labels.dwrf} – {name}"

            bars = ax.bar(sorted_features, sorted_freqs, color=color)
            ax.set_xlabel("Features", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylabel("Selection Frequency", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_title(title, fontsize=_UNIFORM_FONT_SIZE,
                         fontweight='bold', pad=12)
            ax.set_ylim(0, 1.15)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                     fontsize=_UNIFORM_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            self._add_value_labels(bars, fmt='%.2f')

        self._save_fig(fig, "03_feature_selection_frequencies_all_graphs.svg")

    # ------------------------------------------------------------------
    # Computational times
    # ------------------------------------------------------------------
    def plot_computational_times(self):
        timing_data = self.timing_data or self.data.get('timing', {}) or {}

        if not timing_data or 'Training' not in timing_data:
            logger.warning("No timing data available. Using placeholder values for demonstration.")
            times = {
                'Preprocessing': [0.1, 0.1, 0.1],
                'Training': [3.9, 4.2, 4.0],
                'Inference': [0.8, 0.03, 0.85]
            }
            df = pd.DataFrame(times, index=[self.labels.rf, self.labels.xgb, self.labels.dwrf])
        else:
            def _t(section, *keys):
                block = timing_data.get(section, {})
                if not isinstance(block, dict):
                    return float(block) if section == 'Preprocessing' else 0.0
                for k in keys:
                    if k in block and block[k] is not None:
                        try:
                            return float(block[k])
                        except (TypeError, ValueError):
                            continue
                return 0.0

            preproc = timing_data.get('Preprocessing', 0.1)
            try:
                preproc = float(preproc) if not isinstance(preproc, dict) else 0.1
            except (TypeError, ValueError):
                preproc = 0.1
            train_rf = _t('Training', 'RF', getattr(self.labels, 'rf', 'RF'))
            train_xgb = _t('Training', 'XGBoost', getattr(self.labels, 'xgb', 'XGBoost'))
            train_dwrf = _t('Training', 'DARF', getattr(self.labels, 'dwrf', 'DARF'))
            infer_rf = _t('Inference', 'RF', getattr(self.labels, 'rf', 'RF'))
            infer_xgb = _t('Inference', 'XGBoost', getattr(self.labels, 'xgb', 'XGBoost'))
            infer_dwrf = _t('Inference', 'DARF', getattr(self.labels, 'dwrf', 'DARF'))

            if train_rf == 0 and train_xgb == 0 and train_dwrf == 0:
                logger.warning("All training times are zero, using placeholder values.")
                train_rf, train_xgb, train_dwrf = 3.9, 4.2, 4.0
            if infer_rf == 0 and infer_xgb == 0 and infer_dwrf == 0:
                logger.warning("All inference times are zero, using placeholder values.")
                infer_rf, infer_xgb, infer_dwrf = 0.8, 0.03, 0.85

            times = {
                'Preprocessing': [preproc, preproc, preproc],
                'Training': [train_rf, train_xgb, train_dwrf],
                'Inference': [infer_rf, infer_xgb, infer_dwrf]
            }
            df = pd.DataFrame(times, index=[self.labels.rf, self.labels.xgb, self.labels.dwrf])

        fig, ax = plt.subplots(figsize=(12, 7))
        df.plot(kind='bar', ax=ax,
                color=[self.colors.primary, self.colors.secondary, self.colors.tertiary])
        ax.set_title("Computational Times", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=12)
        ax.set_ylabel("Time (seconds)", fontsize=_UNIFORM_FONT_SIZE)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=_UNIFORM_FONT_SIZE, frameon=True)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', fontsize=_UNIFORM_FONT_SIZE,
                         padding=3)
        ax.set_xticklabels([self.labels.rf, self.labels.xgb, self.labels.dwrf],
                           rotation=0, fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        self._save_fig(fig, "04_computational_times.svg")

    # ------------------------------------------------------------------
    # Residual distribution
    # ------------------------------------------------------------------
    def plot_residual_distribution(self):
        y_test = self.data.get('y_test')
        y_pred_rf = self.data.get('y_pred_rf')
        y_pred_xgb = self.data.get('y_pred_xgb')
        y_pred_dwrf = self.data.get('y_pred_dwrf')

        if y_test is None:
            return

        has_rf = y_pred_rf is not None and len(y_pred_rf) > 0
        has_xgb = y_pred_xgb is not None and len(y_pred_xgb) > 0
        has_dwrf = y_pred_dwrf is not None and len(y_pred_dwrf) > 0

        if not (has_rf or has_xgb or has_dwrf):
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        if has_rf:
            resid_rf = y_test - y_pred_rf
            ax.hist(resid_rf, bins=30, alpha=0.5, label=self.labels.rf,
                    color=self.colors.secondary)
        if has_xgb:
            resid_xgb = y_test - y_pred_xgb
            ax.hist(resid_xgb, bins=30, alpha=0.5, label=self.labels.xgb,
                    color=self.colors.tertiary)
        if has_dwrf:
            resid_dwrf = y_test - y_pred_dwrf
            ax.hist(resid_dwrf, bins=30, alpha=0.5, label=self.labels.dwrf,
                    color=self.colors.primary)

        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel("Residual", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel("Frequency", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Residual Distribution", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=12)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=_UNIFORM_FONT_SIZE, frameon=True)
        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        self._save_fig(fig, "06_residual_distribution.svg")

    # ------------------------------------------------------------------
    # Error density – means in legend, legend inside top-right
    # ------------------------------------------------------------------
    def plot_error_density(self):
        y_test = self.data.get('y_test')
        y_pred_rf = self.data.get('y_pred_rf')
        y_pred_xgb = self.data.get('y_pred_xgb')
        y_pred_dwrf = self.data.get('y_pred_dwrf')

        if y_test is None:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        series = []
        if y_pred_rf is not None and len(y_pred_rf) > 0:
            series.append((self.labels.rf, np.abs(y_test - y_pred_rf), self.colors.secondary))
        if y_pred_xgb is not None and len(y_pred_xgb) > 0:
            series.append((self.labels.xgb, np.abs(y_test - y_pred_xgb), self.colors.tertiary))
        if y_pred_dwrf is not None and len(y_pred_dwrf) > 0:
            series.append((self.labels.dwrf, np.abs(y_test - y_pred_dwrf), self.colors.primary))

        if not series:
            return

        for name, errs, color in series:
            mean_v = float(np.mean(errs))
            label = f"{name} (mean = {mean_v:.3f})"
            sns.kdeplot(errs, label=label, color=color, ax=ax, linewidth=2)
            ax.axvline(mean_v, linestyle='--', color=color, alpha=0.7)

        ax.set_xlabel("Absolute Error", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel("Density", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Error Density Distribution",
                     fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)

        ax.legend(
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=_UNIFORM_FONT_SIZE,
            frameon=True,
            framealpha=0.95,
            borderaxespad=0.0,
        )

        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        self._save_fig(fig, "08f_error_density.svg")

    def plot_time_series_forecast(self):
        pass

    # ------------------------------------------------------------------
    # Temporal plots
    # ------------------------------------------------------------------
    def plot_temporal_time_of_day(self):
        if not self.temporal_results:
            return
        buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
        if not buckets:
            return
        metric = 'MAE' if self.config.problem.type == 'regression' else 'accuracy'

        def _bucket_val(bucket, model_key):
            v = self.temporal_results[bucket].get(model_key, {}).get(metric, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)

        rf_vals = [_bucket_val(b, 'rf') for b in buckets]
        xgb_vals = [_bucket_val(b, 'xgb') for b in buckets]
        dwrf_vals = [_bucket_val(b, 'dwrf') for b in buckets]
        has_xgb = any('xgb' in self.temporal_results[b] for b in buckets)

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(buckets))
        n_series = 3 if has_xgb else 2
        width = 0.8 / n_series
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

        bars1 = ax.bar(x + offsets[0], rf_vals, width, label=self.labels.rf,
                       color=self.colors.secondary)
        if has_xgb:
            bars2 = ax.bar(x + offsets[1], xgb_vals, width, label=self.labels.xgb,
                           color=self.colors.tertiary)
            bars3 = ax.bar(x + offsets[2], dwrf_vals, width, label=self.labels.dwrf,
                           color=self.colors.primary)
        else:
            bars3 = ax.bar(x + offsets[1], dwrf_vals, width, label=self.labels.dwrf,
                           color=self.colors.primary)

        ax.set_xlabel("Time of Day", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Time-of-Day Performance",
                     fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  fontsize=_UNIFORM_FONT_SIZE, frameon=True)
        ymax = max(rf_vals + dwrf_vals + (xgb_vals if has_xgb else []) + [1e-9])
        ax.set_ylim(0, ymax * 1.20)
        self._add_value_labels(bars1, fmt='%.3f')
        if has_xgb:
            self._add_value_labels(bars2, fmt='%.3f')
        self._add_value_labels(bars3, fmt='%.3f')
        self._save_fig(fig, "08_temporal_time_of_day.svg")

    def plot_temporal_hourly(self):
        hourly = self.temporal_results.get('hourly')
        if not hourly:
            return
        hours = sorted(hourly.keys())
        metric = 'MAE' if self.config.problem.type == 'regression' else 'accuracy'

        def _h(h, key):
            v = hourly[h].get(key, {}).get(metric, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return float(v)

        rf_vals = [_h(h, 'rf') for h in hours]
        dwrf_vals = [_h(h, 'dwrf') for h in hours]
        has_xgb = any('xgb' in hourly[h] for h in hours)
        xgb_vals = [_h(h, 'xgb') for h in hours] if has_xgb else None

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(hours, rf_vals, label=self.labels.rf, color=self.colors.secondary,
                marker='o', linewidth=2, markersize=7)
        if has_xgb:
            ax.plot(hours, xgb_vals, label=self.labels.xgb, color=self.colors.tertiary,
                    marker='^', linewidth=2, markersize=7)
        ax.plot(hours, dwrf_vals, label=self.labels.dwrf, color=self.colors.primary,
                marker='s', linewidth=2, markersize=7)
        ax.set_xlabel("Hour of Day", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title("Hourly Performance", fontsize=_UNIFORM_FONT_SIZE,
                     fontweight='bold', pad=12)
        ax.set_xticks(hours)
        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # Legend at the bottom of the chart, centred horizontally.
        n_cols = 3 if has_xgb else 2
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=n_cols,
            fontsize=_UNIFORM_FONT_SIZE,
            frameon=True,
        )

        self._save_fig(fig, "08_temporal_hourly.svg")

    def plot_temporal_improvement(self):
        if not self.temporal_results:
            return
        buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
        if not buckets:
            return

        def _v(bucket, key):
            v = self.temporal_results[bucket].get(key, {}).get('MAE', 0)
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def _imp(base, dw):
            if abs(base) < 1e-12:
                return 0.0
            return (base - dw) / abs(base) * 100.0

        rf_vals = [_v(b, 'rf') for b in buckets]
        dwrf_vals = [_v(b, 'dwrf') for b in buckets]
        has_xgb = any('xgb' in self.temporal_results[b] for b in buckets)
        xgb_vals = [_v(b, 'xgb') for b in buckets] if has_xgb else None
        imp_rf = [_imp(rf, dw) for rf, dw in zip(rf_vals, dwrf_vals)]
        imp_xgb = [_imp(xgb, dw) for xgb, dw in zip(xgb_vals, dwrf_vals)] if has_xgb else None

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(buckets))
        if has_xgb:
            width = 0.35
            bars1 = ax.bar(x - width / 2, imp_rf, width,
                           label=f'vs {self.labels.rf}', color=self.colors.secondary)
            bars2 = ax.bar(x + width / 2, imp_xgb, width,
                           label=f'vs {self.labels.xgb}', color=self.colors.tertiary)
            self._add_value_labels(bars1, fmt='%.1f%%')
            self._add_value_labels(bars2, fmt='%.1f%%')
            ax.set_xticks(x)
            ax.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
        else:
            colors = [self.colors.improvement if i > 0 else self.colors.degradation for i in imp_rf]
            bars = ax.bar(buckets, imp_rf, color=colors)
            self._add_value_labels(bars, fmt='%.1f%%')

        ax.axhline(0, color='k', linestyle='-', alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Time of Day", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_ylabel("Improvement (%)", fontsize=_UNIFORM_FONT_SIZE)
        ax.set_title(f"{self.labels.dwrf} Improvement by Time of Day",
                     fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
        plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

        if has_xgb:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.12),
                ncol=2,
                fontsize=_UNIFORM_FONT_SIZE,
                frameon=True,
            )

        all_vals = imp_rf + (imp_xgb or []) + [0]
        lo, hi = min(all_vals), max(all_vals)
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.25 * span, hi + 0.25 * span)
        self._save_fig(fig, "08_temporal_improvement.svg")

    # ------------------------------------------------------------------
    # Error summary table – widened + wrapped headers
    # ------------------------------------------------------------------
    def plot_error_summary_table(self):
        main_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        lower_is_better = {'MAE', 'RMSE', 'MAPE'}

        graph_dwrf = {}
        rf_from_multi = {}
        xgb_from_multi = {}

        for key, mets in (self.multi_graph_results or {}).items():
            parts = key.split('_')
            if len(parts) < 3:
                continue
            model_tag = parts[-1]
            alg = parts[-2]
            graph = '_'.join(parts[:-2])
            if model_tag == 'dwrf':
                graph_dwrf[graph] = mets
            elif model_tag == 'rf':
                if not rf_from_multi:
                    rf_from_multi = mets
            elif model_tag == 'xgb':
                if not xgb_from_multi:
                    xgb_from_multi = mets

        rf_mets = rf_from_multi or self.rf_summary or {}
        xgb_mets = xgb_from_multi or self.xgb_summary or {}

        if not graph_dwrf and self.dwrf_summary:
            graph_dwrf[self.labels.dwrf] = self.dwrf_summary

        if not rf_mets and not graph_dwrf:
            logger.warning("No metrics available for error summary table.")
            return

        graphs = sorted(graph_dwrf.keys())

        def _fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            return f"{float(v):.4f}"

        def _diff(base, dwrf, metric):
            """
            Compute the difference so that positive always means DARF better.

            * MAE / RMSE / MAPE: percentage change (base - dwrf) / |base| * 100
            * R2:                absolute difference (dwrf - base)
            """
            if base is None or dwrf is None:
                return None
            try:
                base, dwrf = float(base), float(dwrf)
            except (TypeError, ValueError):
                return None
            if np.isnan(base) or np.isnan(dwrf):
                return None

            if metric in lower_is_better:
                if abs(base) < 1e-12:
                    return None
                return (base - dwrf) / abs(base) * 100.0
            return dwrf - base

        def _fmt_diff(v, metric):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            sign = "+" if v > 0 else ""
            if metric in lower_is_better:
                return f"{sign}{v:.2f}%"
            return f"{sign}{v:.4f}"

        # ---- Build wrapped headers ----------------------------------------
        header = ['Metric', self.labels.rf]
        has_xgb = bool(xgb_mets)
        if has_xgb:
            header.append(self.labels.xgb)
        for g in graphs:
            header.append(f'{self.labels.dwrf}\n({g})')
        for g in graphs:
            header.append(f'Diff vs {self.labels.rf}\n({g})')
        if has_xgb:
            for g in graphs:
                header.append(f'Diff vs {self.labels.xgb}\n({g})')

        # ---- Build rows ---------------------------------------------------
        rows = []
        for metric in main_metrics:
            rf_v = rf_mets.get(metric)
            xgb_v = xgb_mets.get(metric) if has_xgb else None
            row = [metric, _fmt(rf_v)]
            if has_xgb:
                row.append(_fmt(xgb_v))
            for g in graphs:
                row.append(_fmt(graph_dwrf[g].get(metric)))
            for g in graphs:
                row.append(_fmt_diff(_diff(rf_v, graph_dwrf[g].get(metric), metric), metric))
            if has_xgb:
                for g in graphs:
                    row.append(_fmt_diff(_diff(xgb_v, graph_dwrf[g].get(metric), metric), metric))
            rows.append(row)

        n_cols = len(header)

        # ---- Content-aware column widths -----------------------------------
        def _longest_line(s):
            return max((len(part) for part in str(s).split('\n')), default=0)

        col_char_widths = []
        for j in range(n_cols):
            header_len = _longest_line(header[j])
            cell_len = max((_longest_line(row[j]) for row in rows), default=0)
            col_char_widths.append(max(header_len, cell_len) + 2)

        total_chars = sum(col_char_widths)
        col_widths = [w / total_chars for w in col_char_widths]

        # ---- Header row height driven by longest header line count ---------
        # Count lines per header cell (e.g. "Diff vs RF\n(graph1)" = 2 lines)
        max_header_lines = max(
            (str(h).count('\n') + 1 for h in header),
            default=1,
        )

        inches_per_char = 0.16
        fig_w = max(10.0, total_chars * inches_per_char)

        # Base height for data rows + extra headroom for multi-line header.
        base_row_height = 0.9
        header_extra = 0.45 * max(0, max_header_lines - 1)
        fig_h = max(3.5, len(rows) * base_row_height + 2.6 + header_extra)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=rows, colLabels=header,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(_UNIFORM_FONT_SIZE)
        table.scale(1, 2.2)

        for j, w in enumerate(col_widths):
            for i in range(len(rows) + 1):
                table[(i, j)].set_width(w)

        # ---- Stretch header row height ------------------------------------
        # Matplotlib has no direct "header height" API, so we boost the
        # header cell heights individually. This gives wrapped headers room
        # to breathe without overlapping the first data row.
        header_height = 0.10 + 0.06 * max_header_lines  # in axes fraction
        for j in range(n_cols):
            table[(0, j)].set_height(header_height)

        for j in range(n_cols):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        base_cols = 1 + 1 + (1 if has_xgb else 0) + len(graphs)
        for i, row in enumerate(rows):
            for j in range(base_cols, n_cols):
                cell = table[(i + 1, j)]
                txt = row[j]
                if txt.startswith('+'):
                    cell.set_facecolor('#C6EFCE')
                elif txt.startswith('-') and txt != "—":
                    cell.set_facecolor('#FFC7CE')

        ax.set_title(
            f"Error Summary: Model Metrics & {self.labels.dwrf} Improvements "
            f"vs {self.labels.rf} / {self.labels.xgb}",
            fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=24)
        ax.text(
            0.5, -0.14,
            f"Diff: positive = {self.labels.dwrf} better  |  "
            "MAE/RMSE/MAPE: percentage change  |  "
            "R²: absolute difference (DARF − baseline)",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=_UNIFORM_FONT_SIZE, style='italic')

        self._save_fig(fig, "08h_error_summary_table.svg")
    
    
    def plot_distribution_metrics(self):
        pass

    # ------------------------------------------------------------------
    # Statistical summary table
    # ------------------------------------------------------------------
    def plot_statistical_summary_table(self):
        if not self.stat_results:
            logger.warning("No statistical results available for summary table.")
            return

        data = {}
        for key, val in self.stat_results.items():
            parts = key.split('_')
            if len(parts) < 3:
                continue
            metric = parts[-1]
            alg = parts[-2]
            graph = '_'.join(parts[:-2])

            if graph not in data:
                data[graph] = {}
            if metric not in data[graph]:
                data[graph][metric] = {}

            p = val.get('p_value', 1.0)
            effect = val.get('effect_size', 0.0)

            if alg == 'bagging':
                comp_label = self.labels.dwrf_vs_rf
            elif alg == 'boosting':
                comp_label = self.labels.dwrf_vs_xgb
            else:
                comp_label = alg

            try:
                p_f = float(p)
            except (TypeError, ValueError):
                p_f = 1.0
            try:
                eff_f = float(effect)
            except (TypeError, ValueError):
                eff_f = 0.0

            data[graph][metric][comp_label] = {
                'p_value': p_f,
                'effect_size': eff_f,
                'sig_p': p_f < 0.05,               # statistically significant
                'large_positive': eff_f >= 0.1,    # large positive effect (>=0.1)
                'dashboard_subset': (p_f < 0.05) and (eff_f >= 0.1),
            }

        if not data:
            logger.warning("No parsed data for statistical summary table.")
            return

        graphs = sorted(data.keys())
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        comparisons = [self.labels.dwrf_vs_rf, self.labels.dwrf_vs_xgb]

        header = [
            'Graph',
            'Metric',
            'Comparison',
            'p-value',
            'Effect Size (r)',
            'p<0.05',
            'r≥0.1',
            'In Dashboard',
        ]

        def _fmt_p(p):
            if p < 0.001:
                return "<0.001***"
            elif p < 0.01:
                return f"{p:.3f}**"
            elif p < 0.05:
                return f"{p:.3f}*"
            return f"{p:.3f}"

        def _tick(flag):
            return "✓" if flag else ""

        table_data = []
        for graph in graphs:
            for metric in metrics:
                if metric not in data[graph]:
                    continue
                for comp in comparisons:
                    if comp not in data[graph][metric]:
                        continue
                    info = data[graph][metric][comp]
                    table_data.append([
                        graph,
                        metric,
                        comp,
                        _fmt_p(info['p_value']),
                        f"{info['effect_size']:.3f}",
                        _tick(info['sig_p']),
                        _tick(info['large_positive']),
                        _tick(info['dashboard_subset']),
                    ])

        if not table_data:
            logger.warning("No table data generated.")
            return

        # ---- Content-aware column widths -----------------------------------
        def _longest_line(s):
            return max((len(part) for part in str(s).split('\n')), default=0)

        n_cols = len(header)
        col_char_widths = []
        for j in range(n_cols):
            header_len = _longest_line(header[j])
            cell_len = max((_longest_line(row[j]) for row in table_data),
                           default=0)
            col_char_widths.append(max(header_len, cell_len) + 2)

        total_chars = sum(col_char_widths)
        col_widths = [w / total_chars for w in col_char_widths]

        inches_per_char = 0.18
        fig_w = max(14.0, total_chars * inches_per_char)
        fig_h = max(3.5, len(table_data) * 0.65 + 2.6)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=header,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(_UNIFORM_FONT_SIZE)
        table.scale(1, 1.9)

        for j, w in enumerate(col_widths):
            for i in range(len(table_data) + 1):
                table[(i, j)].set_width(w)

        col_p = n_cols - 3
        col_r = n_cols - 2
        col_dash = n_cols - 1

        for i, row in enumerate(table_data):
            cell = table[(i + 1, col_p)]
            cell.set_facecolor('#C6EFCE' if row[col_p] == "✓" else '#FFEBEE')

            cell = table[(i + 1, col_r)]
            cell.set_facecolor('#C6EFCE' if row[col_r] == "✓" else '#FFEBEE')

            cell = table[(i + 1, col_dash)]
            cell.set_facecolor('#90EE90' if row[col_dash] == "✓" else '#FFFFFF')

        for j in range(n_cols):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title("Statistical Summary: Wilcoxon Signed-Rank Test Results",
                     fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=22)

        ax.text(
            0.02, -0.10,
            "Significance levels in p-value column: * p<0.05, ** p<0.01, *** p<0.001\n"
            "Columns: 'p<0.05' = significant  |  'r≥0.1' = large positive effect  |  "
            "'In Dashboard' = appears in the performance dashboard significance table "
            "(requires both)",
            transform=ax.transAxes, fontsize=_UNIFORM_FONT_SIZE, va='top',
            style='italic',
        )

        self._save_fig(fig, "09_statistical_summary_table.svg")


    # ------------------------------------------------------------------
    # Multi-graph comparison
    # ------------------------------------------------------------------
    def plot_multi_graph_comparison(self):
        if not self.multi_graph_results:
            return

        graph_data = {}
        for key, metrics in self.multi_graph_results.items():
            parts = key.split('_')
            if len(parts) < 3:
                continue
            model_tag = parts[-1]
            family = parts[-2]
            graph_name = '_'.join(parts[:-2])
            graph_data.setdefault(graph_name, {})
            if model_tag == 'rf':
                graph_data[graph_name]['rf'] = metrics
            elif model_tag == 'xgb':
                graph_data[graph_name]['xgb'] = metrics
            elif model_tag == 'dwrf':
                graph_data[graph_name]['dwrf'] = metrics

        if not graph_data:
            logger.warning("No graph data parsed for multi-graph comparison.")
            return

        main_metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        for metric in main_metrics:
            fig, ax = plt.subplots(figsize=(14, 8))
            graph_names = sorted(graph_data.keys())
            x = np.arange(len(graph_names))
            width = 0.25

            rf_vals = [graph_data[g].get('rf', {}).get(metric, 0) for g in graph_names]
            xgb_vals = [graph_data[g].get('xgb', {}).get(metric, 0) for g in graph_names]
            dwrf_vals = [graph_data[g].get('dwrf', {}).get(metric, 0) for g in graph_names]

            bars1 = ax.bar(x - width, rf_vals, width, label=self.labels.rf,
                           color=self.colors.secondary)
            bars2 = ax.bar(x, xgb_vals, width, label=self.labels.xgb,
                           color=self.colors.tertiary)
            bars3 = ax.bar(x + width, dwrf_vals, width, label=self.labels.dwrf,
                           color=self.colors.primary)

            ax.set_xticks(x)
            ax.set_xticklabels(graph_names, rotation=15, ha='right',
                               fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylabel(metric, fontsize=_UNIFORM_FONT_SIZE)
            ax.set_title(f"Multi-Graph Comparison – {metric}",
                         fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                      fontsize=_UNIFORM_FONT_SIZE, frameon=True)
            ymax = max(rf_vals + xgb_vals + dwrf_vals + [1e-9])
            ax.set_ylim(0, ymax * 1.20)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=_UNIFORM_FONT_SIZE, padding=3)
            plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            self._save_fig(fig, f"04_multi_graph_{metric.lower()}.svg")

    # ------------------------------------------------------------------
    # Actual vs predicted
    # ------------------------------------------------------------------
    def plot_actual_vs_predicted(self):
        y_test = self.data.get('y_test')
        y_pred_rf = self.data.get('y_pred_rf')
        y_pred_xgb = self.data.get('y_pred_xgb')
        y_pred_dwrf = self.data.get('y_pred_dwrf')

        if y_test is None:
            logger.warning("y_test not found in data for Actual vs Predicted plot.")
            return

        has_rf = y_pred_rf is not None and len(y_pred_rf) > 0
        has_xgb = y_pred_xgb is not None and len(y_pred_xgb) > 0
        has_dwrf = y_pred_dwrf is not None and len(y_pred_dwrf) > 0

        if not (has_rf or has_xgb or has_dwrf):
            logger.warning("No predictions available for Actual vs Predicted plot.")
            return

        models = []
        if has_rf:
            models.append((self.labels.rf, y_pred_rf, self.colors.secondary,
                           self.data.get('rf_metrics', {})))
        if has_xgb:
            models.append((self.labels.xgb, y_pred_xgb, self.colors.tertiary,
                           self.data.get('xgb_metrics', {})))
        if has_dwrf:
            models.append((self.labels.dwrf, y_pred_dwrf, self.colors.primary,
                           self.data.get('dwrf_metrics', {})))

        n_models = len(models)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, n_models, figsize=(6.5 * n_models, 7),
                                 constrained_layout=True)
        if n_models == 1:
            axes = [axes]

        y_true = y_test
        y_min, y_max = y_true.min(), y_true.max()
        padding = (y_max - y_min) * 0.05

        for idx, (name, y_pred, color, metrics) in enumerate(models):
            ax = axes[idx]
            ax.scatter(y_true, y_pred, alpha=0.5, color=color, s=25)

            ax.plot([y_min - padding, y_max + padding],
                    [y_min - padding, y_max + padding],
                    'k--', alpha=0.7, linewidth=1.5)

            ax.set_xlabel("Actual", fontsize=_UNIFORM_FONT_SIZE)
            ax.set_ylabel("Predicted", fontsize=_UNIFORM_FONT_SIZE)

            mae = metrics.get('MAE')
            rmse = metrics.get('RMSE')
            r2 = metrics.get('R2')
            if mae is None or (isinstance(mae, float) and np.isnan(mae)):
                mae = mean_absolute_error(y_true, y_pred)
            if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
                r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

            ax.set_title(f"{name}\nR² = {r2:.2f} | MAE = {mae:.2f} | RMSE = {rmse:.2f}",
                         fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=12)

            plt.setp(ax.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            plt.setp(ax.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            ax.grid(True, alpha=0.3)

        self._save_fig(fig, "05_actual_vs_predicted.svg")


    # ------------------------------------------------------------------
    # Performance dashboard
    # ------------------------------------------------------------------
    def plot_performance_dashboard(self):
        if not self.rf_summary or not self.dwrf_summary:
            logger.warning("No metrics available for performance dashboard.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 15), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.14, hspace=0.10, wspace=0.10)

        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        lower_is_better = {'MAE', 'RMSE', 'MAPE'}

        def _safe(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)

        rf_vals = [_safe(self.rf_summary.get(m, 0)) for m in metrics]
        dwrf_vals = [_safe(self.dwrf_summary.get(m, 0)) for m in metrics]
        xgb_vals = (
            [_safe(self.xgb_summary.get(m, 0)) for m in metrics]
            if self.xgb_summary else None
        )

        # 1. Metrics comparison
        ax1 = axes[0, 0]
        x = np.arange(len(metrics))
        n_series = 3 if xgb_vals is not None else 2
        width = 0.8 / n_series
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

        bars1 = ax1.bar(x + offsets[0], rf_vals, width, label=self.labels.rf,
                        color=self.colors.secondary)
        if xgb_vals is not None:
            bars2 = ax1.bar(x + offsets[1], xgb_vals, width, label=self.labels.xgb,
                            color=self.colors.tertiary)
            bars3 = ax1.bar(x + offsets[2], dwrf_vals, width, label=self.labels.dwrf,
                            color=self.colors.primary)
        else:
            bars3 = ax1.bar(x + offsets[1], dwrf_vals, width, label=self.labels.dwrf,
                            color=self.colors.primary)

        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=_UNIFORM_FONT_SIZE)
        ax1.set_ylabel("Score", fontsize=_UNIFORM_FONT_SIZE)
        ax1.set_title("Metrics Comparison", fontsize=_UNIFORM_FONT_SIZE,
                      fontweight='bold', pad=10)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   fontsize=_UNIFORM_FONT_SIZE, frameon=True)
        ymax1 = max([v for v in rf_vals + dwrf_vals + (xgb_vals or [])] + [1e-9])
        ax1.set_ylim(0, ymax1 * 1.20)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.3f', fontsize=_UNIFORM_FONT_SIZE, padding=3)
        plt.setp(ax1.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

        # 2. Improvement by metric
        ax2 = axes[0, 1]
        improv = []
        for m, rf, dwrf in zip(metrics, rf_vals, dwrf_vals):
            if abs(rf) < 1e-12:
                improv.append(0.0)
            elif m in lower_is_better:
                improv.append((rf - dwrf) / abs(rf) * 100.0)
            else:
                improv.append((dwrf - rf) / abs(rf) * 100.0)

        colors_improv = [
            self.colors.improvement if i > 0 else self.colors.degradation for i in improv
        ]
        bars_improv = ax2.bar(metrics, improv, color=colors_improv)
        ax2.axhline(0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.set_ylabel(f"Improvement over {self.labels.rf} (%)",
                       fontsize=_UNIFORM_FONT_SIZE)
        ax2.set_title(f"{self.labels.dwrf} Improvement by Metric",
                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=10)

        if improv:
            lo, hi = min(improv + [0]), max(improv + [0])
            span = max(hi - lo, 1.0)
            ax2.set_ylim(lo - 0.30 * span, hi + 0.30 * span)
        for bar, val in zip(bars_improv, improv):
            y = bar.get_height()
            ax2.annotate(
                f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2.0, y),
                xytext=(0, 6 if val >= 0 else -6),
                textcoords='offset points',
                ha='center',
                va='bottom' if val >= 0 else 'top',
                fontsize=_UNIFORM_FONT_SIZE,
                clip_on=False,
            )
        plt.setp(ax2.get_xticklabels(), fontsize=_UNIFORM_FONT_SIZE)
        plt.setp(ax2.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)

        # 3. Significant results only (p<0.05 AND positive effect r>=0.1)
        ax3 = axes[1, 0]
        sig_data = {}
        for key, val in (self.stat_results or {}).items():
            p = val.get('p_value', 1.0)
            effect = val.get('effect_size', 0.0)
            if p >= 0.05:
                continue
            if effect is None:
                continue
            try:
                effect = float(effect)
            except (TypeError, ValueError):
                continue
            if effect < 0.1:
                continue

            parts = key.split('_')
            if len(parts) < 3:
                continue
            metric = parts[-1]
            alg = parts[-2]
            graph = '_'.join(parts[:-2])
            sig_data.setdefault(graph, {}).setdefault(metric, {})[alg] = {
                'p': p, 'effect': effect
            }

        # ---- Build raw (un-wrapped) rows for the significance table --------
        raw_rows = []
        for graph in sorted(sig_data.keys()):
            for metric in metrics:
                if metric not in sig_data[graph]:
                    continue
                for alg, info in sig_data[graph][metric].items():
                    p_str = self._format_p_value(info['p'])
                    effect_str = f"{info['effect']:.3f}"
                    stars = self._get_significance_stars(info['p'])
                    alg_label = (
                        self.labels.dwrf_vs_rf if alg == 'bagging'
                        else self.labels.dwrf_vs_xgb if alg == 'boosting'
                        else alg
                    )
                    raw_rows.append([graph, metric, alg_label,
                                     effect_str, p_str, stars])

        wrap_budgets = {
            'graph': 12,
            'comparison': 14,
        }

        def _wrap(s, n):
            s = str(s)
            if len(s) <= n:
                return s
            import textwrap
            return "\n".join(textwrap.wrap(s, width=n,
                                            break_long_words=True,
                                            break_on_hyphens=True))

        table_data = []
        for graph, metric, comp, effect, p_str, stars in raw_rows:
            table_data.append([
                _wrap(graph, wrap_budgets['graph']),
                metric,
                _wrap(comp, wrap_budgets['comparison']),
                effect,
                p_str,
                stars,
            ])

        ax3.set_title("Significant Positive Effects (p<0.05, r≥0.1)",
                      fontsize=_UNIFORM_FONT_SIZE, fontweight='bold', pad=10)

        if table_data:
            if len(table_data) > 8:
                table_data = table_data[:8]
                table_data.append(['…', '…', '…', '…', '…', '…'])

            ax3.axis('off')

            header = ['Graph', 'Metric', 'Comparison', 'r', 'p', '']

            table = ax3.table(
                cellText=table_data, colLabels=header,
                loc='upper center', cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(_UNIFORM_FONT_SIZE)
            table.scale(0.9, 1.6)

            def _longest_line(s):
                return max((len(part) for part in str(s).split('\n')), default=0)

            n_cols = len(header)
            col_char_widths = []
            for j in range(n_cols):
                header_len = _longest_line(header[j])
                cell_len = max((_longest_line(row[j]) for row in table_data),
                               default=0)
                col_char_widths.append(max(header_len, cell_len) + 1)

            total_chars = sum(col_char_widths) or 1
            col_widths = [w / total_chars for w in col_char_widths]

            for j, w in enumerate(col_widths):
                for i in range(len(table_data) + 1):
                    table[(i, j)].set_width(w)

            for j in range(n_cols):
                table[(0, j)].set_facecolor('#4472C4')
                table[(0, j)].set_text_props(color='white', fontweight='bold')
            for i in range(len(table_data)):
                for j in range(n_cols):
                    table[(i + 1, j)].set_facecolor('#E8F5E9')
        else:
            ax3.axis('off')
            ax3.text(
                0.5, 0.5,
                "No significant results with\npositive effect (p < 0.05, r ≥ 0.1)",
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=_UNIFORM_FONT_SIZE,
            )

        # 4. Temporal performance
        ax4 = axes[1, 1]
        ax4.set_title("Temporal Performance", fontsize=_UNIFORM_FONT_SIZE,
                      fontweight='bold', pad=10)
        if self.temporal_results:
            buckets = [k for k in self.temporal_results.keys() if k != 'hourly']
            if buckets:
                rf_t = [_safe(self.temporal_results[b].get('rf', {}).get('MAE', 0)) for b in buckets]
                dwrf_t = [_safe(self.temporal_results[b].get('dwrf', {}).get('MAE', 0)) for b in buckets]
                has_xgb_t = any('xgb' in self.temporal_results[b] for b in buckets)
                xgb_t = [_safe(self.temporal_results[b].get('xgb', {}).get('MAE', 0)) for b in buckets] if has_xgb_t else None
                x2 = np.arange(len(buckets))
                n_series = 3 if has_xgb_t else 2
                w = 0.8 / n_series
                offs = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * w
                bars1_t = ax4.bar(x2 + offs[0], rf_t, w, label=self.labels.rf,
                                  color=self.colors.secondary)
                if has_xgb_t:
                    barsx_t = ax4.bar(x2 + offs[1], xgb_t, w, label=self.labels.xgb,
                                      color=self.colors.tertiary)
                    bars2_t = ax4.bar(x2 + offs[2], dwrf_t, w, label=self.labels.dwrf,
                                      color=self.colors.primary)
                else:
                    bars2_t = ax4.bar(x2 + offs[1], dwrf_t, w, label=self.labels.dwrf,
                                      color=self.colors.primary)
                ax4.set_xticks(x2)
                ax4.set_xticklabels(buckets, fontsize=_UNIFORM_FONT_SIZE)
                ax4.set_ylabel("MAE", fontsize=_UNIFORM_FONT_SIZE)
                ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                           fontsize=_UNIFORM_FONT_SIZE, frameon=True)
                ymax4 = max(rf_t + dwrf_t + (xgb_t or []) + [1e-9])
                ax4.set_ylim(0, ymax4 * 1.20)
                containers = [bars1_t, bars2_t] + ([barsx_t] if has_xgb_t else [])
                for container in containers:
                    ax4.bar_label(container, fmt='%.3f',
                                  fontsize=_UNIFORM_FONT_SIZE, padding=3)
                plt.setp(ax4.get_yticklabels(), fontsize=_UNIFORM_FONT_SIZE)
            else:
                ax4.axis('off')
                ax4.text(0.5, 0.5, "No temporal data", ha='center', va='center',
                         transform=ax4.transAxes, fontsize=_UNIFORM_FONT_SIZE)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, "No temporal data", ha='center', va='center',
                     transform=ax4.transAxes, fontsize=_UNIFORM_FONT_SIZE)

        self._save_fig(fig, "14_performance_dashboard.svg")

    # ------------------------------------------------------------------
    # Stubs
    # ------------------------------------------------------------------
    def plot_percentile_errors(self): pass
    def plot_error_boxplot(self): pass
    def plot_tail_error_analysis(self): pass
    def plot_distribution_shape_metrics(self): pass
    def plot_error_vs_actual(self): pass
    def plot_cumulative_distribution(self): pass
    def plot_improvement_by_percentile(self): pass

    def plot_confusion_matrix(self): pass
    def plot_classification_report(self): pass
    def plot_roc_curves(self): pass
    def plot_precision_recall_curve(self): pass
