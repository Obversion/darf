# DARF V1 – Domain-Aware Random Forest

## Overview

**DARF (Domain-Aware Random Forest)** is a Python framework for integrating domain knowledge into ensemble machine learning models through a graph-based prior weighting mechanism.

DARF supports Random Forest (bagging) and is designed for reproducible experimentation, sensitivity analysis, statistical evaluation, and model deployment.

### Key Features

* Regression
* Random Forest and XGBoost ensemble model Baselines
* Up to **3 expert knowledge graphs** for sensitivity analysis
* Domain Prior Weights (DPW) estimated using Ridge Regression and dynamic programming
* Sensitivity analysis for smoothing parameter **α** and Ridge penalty **λ**
* Repeated experiments with configurable random seeds
* Statistical significance testing, effect sizes, and multiple-testing correction
* Temporal performance analysis, including time-of-day evaluation
* SVG visualisations
* Chronological cross-validation
* Automatic saving of models, predictions, results, configurations, and analysis outputs

---

## Installation

Clone or download the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Optionally install DARF in development mode:

```bash
pip install -e .
```

> **Note:** PATH warnings for scripts such as `onnxruntime_test.exe` can generally be ignored. Use `--no-warn-script-location` if required.

---

## Configuration

All experiment settings are defined in:

```text
config/config.yaml
```

The main configuration sections are:

| Section         | Purpose                                                       |
| --------------- | ------------------------------------------------------------- |
| `experiment`    | Experiment metadata and random seeds                          |
| `visualization` | Plot colours, fonts, styles, and output format                |
| `problem`       | Regression or classification settings                         |
| `dataset`       | Dataset path, splitting, missing values, and cross-validation |
| `features`      | Input features, target, lags, and cyclical features           |
| `graphs`        | Expert domain-knowledge graphs                                |
| `ridge`         | Ridge Regression and λ sensitivity settings                   |
| `domain_prior`  | Domain Prior Weight calculation                               |
| `sampling`      | Feature-sampling and α sensitivity settings                   |
| `algorithm`     | Random Forest and XGBoost configuration                       |
| `evaluation`    | Metrics, statistical tests, and temporal analysis             |
| `output`        | Output directories and files                                  |
| `logging`       | Logging level and output settings                             |

### Problem Type

DARF supports:

```yaml
problem:
  type: regression
```

### Dataset and Cross-Validation

The dataset configuration supports:

* Chronological train/validation/test splits
* Lag feature generation
* Cyclical feature encoding

Example:

```yaml
dataset:
  path: "pv_data.csv"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  cv_type: "chronological"
  n_folds: 5
```

### Domain Knowledge Graphs

Up to three directed acyclic graphs (**DAGs**) can be defined for expert knowledge and sensitivity analysis.

Example:

```yaml
graphs:
  graph_1:
    name: "Expert Graph 1"
    description: "Domain knowledge graph"
    edges:
      Target_Variable: ["Variable1", "Variable2", "Variable3"]
      Variable4: ["Variable2", "Variable6"]
```

### Domain Prior Weights

DARF calculates **Domain Prior Weights (DPW)** using domain knowledge combined with statistical information from Ridge Regression.

The intrinsic influence of a feature can be configured using either an R²-based approach or a fixed value.

```yaml
domain_prior:
  min_intrinsic_influence: 0.01
  use_r_squared_intrinsic: true
  fixed_intrinsic_influence: 0.3
```

### α and λ Sensitivity Analysis

DARF supports sensitivity analysis for:

* **α** — feature-sampling smoothing
* **λ** — Ridge Regression regularisation

Example:

```yaml
sampling:
  smoothing_alpha: 0.2
  alpha_sensitivity:
    enabled: true
    values: [0.0, 0.1, 0.2, 0.3, 0.5]

ridge:
  alpha: 1.0
  lambda_sensitivity:
    enabled: true
    values: [0.01, 0.1, 1.0, 10.0]
```

The feature-sampling probability is determined from the domain prior weights and smoothing parameter.

---

## Algorithms

DARF can run:

### Bagging

Random Forest with configurable:

* Number of estimators
* Maximum depth
* Minimum samples per split/leaf
* Feature sampling
* Class weighting
* Parallel processing

### Boosting

XGBoost with configurable:

* Number of estimators
* Maximum depth
* Learning rate
* Subsampling
* Column sampling
* Regularisation
* Early stopping
* Objective and evaluation metric

Select the desired experiment type:

```yaml
algorithm:
  experiment_type: "comparison"
```

Available options:

```text
bagging      # Random Forest only
boosting     # XGBoost only
comparison   # Random Forest + XGBoost
```

---

## Evaluation

DARF supports repeated experiments using configurable random seeds.

Default settings include:

```yaml
evaluation:
  n_runs: 30
  random_seed_base: 42
  significance_level: 0.05
  correction_method: "none"
```

### Regression Metrics

* MAE
* RMSE
* MAPE
* R²

Evaluation can also include:

* Statistical significance tests
* Effect sizes

### Temporal Analysis

Performance can be evaluated across configurable time-of-day buckets, for example:

```yaml
temporal:
  enabled: true
  time_buckets:
    morning: [6, 11]
    afternoon: [12, 17]
    evening: [18, 23]
```

---

## Running an Experiment

Place the dataset in the configured location and update:

```text
config/config.yaml
```

Then run:

```bash
python main.py --config config/config.yaml
```

To override the output directory:

```bash
python main.py \
  --config config/config.yaml \
  --output /path/to/output
```

For Spyder or other IDE workflows, the repository also includes:

```text
run_darf.py
```


## Outputs

Results are stored under the configured output directory:

```text
darf_output/
├── plots/
│   └── *.svg
├── models/
│   └── *.onnx / *.pkl
├── predictions/
│   └── *.csv
├── results_summary.json
└── config.yaml
```

Depending on the configuration, DARF can produce:

* **15+ plots**
* Aggregated experiment results
* Sensitivity analysis results
* Temporal analysis results
* Test predictions
* Trained models
* A copy of the configuration used for the experiment

---

## Customisation

### Visualisation

Modify the `visualization` section to customise:

* Colours
* Fonts
* Figure resolution
* Plot themes
* Output formats

### Expert Knowledge

Define one, two, or three domain knowledge graphs. DARF automatically evaluates and compares the configured graphs.

### Hyperparameter Sensitivity

Enable `alpha_sensitivity` and `lambda_sensitivity` to evaluate the effect of different α and λ values.

### Temporal Analysis

If the dataset contains an `Hour` feature, temporal analysis can be used to evaluate model performance across different times of day.

---

## Reproducibility

DARF is designed for reproducible experiments through configurable random seeds.

A base seed is defined in the configuration, while repeated evaluation runs use incremented seeds:

```yaml
experiment:
  random_seed: 42

evaluation:
  random_seed_base: 42
  n_runs: 30
```

The configuration used for each experiment is also saved with the results.

---

## Project Structure

```text
DARF/
├── config/
│   └── config.yaml
├── models/
├── plots/
├── predictions/
├── main.py
├── run_darf.py
├── requirements.txt
└── README.md
```

---

## License

See the repository license for terms of use.

---

## Support

For questions, issues, or contributions, please use the repository's GitHub issue tracker or contact the **DARF Research Team**.
