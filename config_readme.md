DWRF V4 – Domain-Weighted Random Forest Framework
Overview
DWRF (Domain-Weighted Random Forest) is a Python framework that integrates domain knowledge into ensemble learning (Random Forest and XGBoost) using a graph-based prior weighting mechanism. It supports:

Regression and Classification

Multiple expert knowledge graphs (up to 3) for sensitivity analysis

Bagging (Random Forest) and Boosting (XGBoost) ensembles

Domain Prior Weights (DPW) computed via Ridge Regression and dynamic programming

Comprehensive evaluation with effect sizes, statistical tests, and multiple testing correction

Temporal performance analysis (e.g., time-of-day, hourly)

Sensitivity analysis for hyperparameters α (smoothing) and λ (ridge penalty)

Publication-quality SVG charts (30+ figures)

ONNX export for deployment (e.g., ML.NET)

Installation
Clone or download the project.

Install dependencies (one‑liner):

bash
pip install -r requirements.txt
If you see PATH warnings about scripts like onnxruntime_test.exe, you can safely ignore them. Use --no-warn-script-location if you wish.

(Optional) Install the package in development mode:

bash
pip install -e .
Configuration – config/config.yaml
All experiment settings are defined in a single YAML file. Below we describe every section and its available options.

1. experiment
General metadata and reproducibility.

Key	Type	Default	Description
name	string	"DWRF-V4"	Name of the experiment
version	string	"4.0.0"	Version tag
description	string	"Domain-Weighted Random Forest V4"	Short description
random_seed	integer	42	Global random seed for reproducibility
2. visualization
Controls the appearance and output of plots.

Sub‑sections:

colors – hex color codes for DWRF, RF, XGBoost, improvement, degradation, node colours, etc.

fonts – font sizes for titles, axis labels, legends.

styles – plot_theme (seaborn theme), figure_dpi (resolution), save_format (svg, png, etc.)

All are fully customisable.

3. problem
Defines the type of task.

Key	Type	Default	Description
type	string	"regression"	"regression" or "classification"
num_classes	integer or null	null	Required for classification (number of classes)
4. dataset
Data loading and cross‑validation settings.

Key	Type	Default	Description
path	string	"pv_data.csv"	Path to CSV dataset
datetime_column	string or null	null	Column name for datetime (used for sorting, lag features)
forecast_horizon	integer	48	(Not used internally; reserved)
train_ratio	float	0.70	Fraction of data for training
val_ratio	float	0.15	Fraction for validation
test_ratio	float	0.15	Fraction for testing
missing_strategy	string	"drop"	How to handle missing values: "drop", "mean", "median", "most_frequent", "constant"
missing_fill_value	any	0	Fill value when missing_strategy = "constant"
cv_type	string	"chronological"	Type of cross‑validation: "chronological" (train/val/test split), "blocked" (non‑overlapping blocks), "rolling" (expanding window)
n_folds	integer	5	Number of folds for blocked/rolling CV
test_window	integer	24	Window size for rolling CV
step_size	integer	12	Step size between folds for rolling CV
5. features
Specifies the feature columns and target.

Key	Type	Default	Description
columns	list of strings	[...]	Names of feature columns
target	string	"PVPower"	Name of target column
lags	list of integers	[]	Lag values to create (e.g., [1, 24] adds shifted columns)
cyclical_features	list of strings	["Hour", "Month"]	Columns to encode cyclically (sin/cos transformation)
6. graphs
Defines up to 3 expert knowledge graphs for sensitivity analysis.

Each graph (graph_1, graph_2, graph_3) is a dictionary with:

Key	Type	Description
name	string	Display name
description	string	Short description
edges	dictionary	Key = parent node, value = list of children (directed edges)
Example:

yaml
edges:
  DC_Power: ["GHI", "Panel_Temp", "Ambient_Temp"]
  Panel_Temp: ["GHI", "Ambient_Temp"]
All graphs must be acyclic (DAG).

7. ridge
Settings for Ridge Regression (DRW estimation).

Key	Type	Default	Description
alpha	float	1.0	Regularisation strength (λ)
fit_intercept	boolean	true	Whether to fit an intercept
normalize	boolean	false	Whether to normalise features
lambda_sensitivity	dictionary	{enabled: true, values: [...]}	Enable/disable λ sensitivity analysis, and list of values to test
8. domain_prior
Controls DPW calculation.

Key	Type	Default	Description
min_intrinsic_influence	float	0.01	Minimum self‑influence (γ) for any node
use_r_squared_intrinsic	boolean	true	If true, γ = 1 − R²; otherwise use fixed_intrinsic_influence
fixed_intrinsic_influence	float	0.3	Constant γ if use_r_squared_intrinsic = false
9. sampling
Feature sampling (smoothing) settings for DWRF.

Key	Type	Default	Description
smoothing_alpha	float	0.2	Smoothing factor (α) for feature sampling: P(f) = (1−α)·DPW[f] + α/p
min_sampling_prob	float	0.001	Minimum sampling probability for any feature
alpha_sensitivity	dictionary	{enabled: true, values: [...]}	Enable/disable α sensitivity, and list of values to test
10. algorithm
Selects which ensemble methods to run and their hyperparameters.

Key	Type	Default	Description
experiment_type	string	"comparison"	"bagging" (only RF), "boosting" (only XGBoost), or "comparison" (both)
bagging	dictionary	(see below)	Random Forest hyperparameters
boosting	dictionary	(see below)	XGBoost hyperparameters
Bagging options:

enabled (bool) – whether to include

n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, n_jobs, class_weight

Boosting options:

enabled (bool)

n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda, n_jobs, objective, eval_metric, early_stopping_rounds

All values are passed directly to sklearn or XGBoost.

11. evaluation
Metrics, statistical tests, and temporal analysis settings.

Key	Type	Default	Description
n_runs	integer	30	Number of repeated runs (with different seeds)
random_seed_base	integer	42	Base seed; each run increments it
significance_level	float	0.05	Alpha for statistical tests
correction_method	string	"bonferroni"	Multiple testing correction: "bonferroni", "holm", or "none"
mape_threshold	float	0.05	(Reserved)
regression_metrics	list	["MAE","RMSE","MAPE","R2"]	Metrics computed for regression
classification_metrics	list	["accuracy","f1_macro",...]	Metrics computed for classification
temporal	dictionary	(see below)	Time‑of‑day analysis settings
Temporal sub‑keys:

enabled (bool) – enable/disable

time_buckets – dictionary mapping bucket names to [start_hour, end_hour] (e.g., morning: [6, 11])

12. export
ONNX export options.

Key	Type	Default	Description
enabled	boolean	false	Enable export
format	string	"onnx"	Export format (currently only ONNX)
output_dir	string	"./models"	Directory to save the exported model
onnx	dictionary	{opset_version: 14, target_platform: "mlnet"}	ONNX‑specific settings
13. output
Where results are saved.

Key	Type	Default	Description
dir	string	"./dwrf_output"	Root output directory
save_plots	boolean	true	Save SVG plots
save_models	boolean	true	Save trained models (pickle)
save_predictions	boolean	true	Save predictions as CSV
save_config	boolean	true	Save a copy of the used config
save_sensitivity_results	boolean	true	Save sensitivity analysis results
save_temporal_results	boolean	true	Save temporal analysis results
14. logging
Logging behaviour.

Key	Type	Default	Description
level	string	"INFO"	Log level (DEBUG, INFO, WARNING, etc.)
log_file	string	"dwrf_run.log"	Path to log file
console_output	boolean	true	Print logs to console
verbose_metrics	boolean	true	Print detailed metrics during evaluation
Running an Experiment
Place your dataset (CSV) and update the configuration file accordingly. Then run:

bash
python main.py --config config/config.yaml
You can override the output directory with:

bash
python main.py --config config/config.yaml --output /path/to/output
For Spyder/IDE users, there is a run_dwrf.py script with the same effect.

Outputs
All results are saved under the specified output.dir:

plots/ – SVG figures (30+ charts)

results_summary.json – aggregated metrics, sensitivities, temporal results

models/ – exported ONNX models (if enabled)

predictions/ – test predictions (if enabled)

config.yaml – copy of the used configuration

Customisation Tips
Colours: Modify the visualization.colors section to match your publication style.

Graphs: You can define 1, 2, or 3 graphs. The framework automatically runs all and compares them.

Hyperparameter sweeps: Enable lambda_sensitivity and alpha_sensitivity to test multiple values; the results will appear in the sensitivity plots.

Temporal analysis: If your data contains a column named "Hour", the framework will automatically create the time‑of‑day plots.

Support
For questions or issues, please contact the DWRF Research Team.

