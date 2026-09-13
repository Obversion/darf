import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum

class ProblemType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class AlgorithmType(Enum):
    BAGGING = "bagging"
    BOOSTING = "boosting"
    COMPARISON = "comparison"

class CVType(Enum):
    CHRONOLOGICAL = "chronological"
    BLOCKED = "blocked"
    ROLLING = "rolling"

@dataclass
class GraphConfig:
    name: str = ""
    description: str = ""
    edges: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class VisualizationColorsConfig:
    primary: str = "#3498db"
    secondary: str = "#e74c3c"
    tertiary: str = "#2ecc71"
    improvement: str = "#27ae60"
    degradation: str = "#e74c3c"
    target_node: str = "#e74c3c"
    root_node: str = "#3498db"
    intermediate_node: str = "#2ecc71"
    background: str = "#ffffff"
    grid: str = "#f0f0f0"

@dataclass
class VisualizationFontsConfig:
    title_size: int = 14
    subtitle_size: int = 12
    axis_label_size: int = 11
    legend_size: int = 10

@dataclass
class VisualizationStylesConfig:
    plot_theme: str = "seaborn-v0_8-darkgrid"
    figure_dpi: int = 150
    save_format: str = "svg"

@dataclass
class VisualizationLabelsConfig:
    rf: str = "RF"
    dwrf: str = "DWRF"
    xgb: str = "XGBoost"
    rf_baseline: str = "RF (unweighted)"
    dwrf_vs_rf: str = "DWRF vs RF"
    dwrf_vs_xgb: str = "DWRF vs XGBoost"


@dataclass
class VisualizationConfig:
    colors: VisualizationColorsConfig = field(default_factory=VisualizationColorsConfig)
    fonts: VisualizationFontsConfig = field(default_factory=VisualizationFontsConfig)
    styles: VisualizationStylesConfig = field(default_factory=VisualizationStylesConfig)
    labels: VisualizationLabelsConfig = field(default_factory=VisualizationLabelsConfig)

@dataclass
class ExperimentConfig:
    name: str = "DWRF-V4"
    version: str = "4.0.0"
    description: str = "Domain-Weighted Random Forest V4"
    random_seed: int = 42

@dataclass
class ProblemConfig:
    type: str = "regression"
    num_classes: Optional[int] = None

@dataclass
class DatasetConfig:
    path: str = "pv_data.csv"
    datetime_column: Optional[str] = None
    forecast_horizon: int = 0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    missing_strategy: str = "drop"
    missing_fill_value: int = 0
    cv_type: Union[str, CVType] = CVType.CHRONOLOGICAL  # Accept both
    n_folds: int = 5
    test_window: int = 24
    step_size: int = 12
    
    def __post_init__(self):
        """Convert cv_type to enum if it's a string."""
        if isinstance(self.cv_type, str):
            try:
                self.cv_type = CVType(self.cv_type)
            except ValueError:
                # Try lowercase matching
                for enum_member in CVType:
                    if enum_member.value == self.cv_type.lower():
                        self.cv_type = enum_member
                        break
                else:
                    raise ValueError(f"Unknown CV type: {self.cv_type}")

@dataclass
class FeaturesConfig:
    columns: List[str] = field(default_factory=list)
    target: str = ""
    lags: List[int] = field(default_factory=list)
    cyclical_features: List[str] = field(default_factory=list)

@dataclass
class RidgeConfig:
    alpha: float = 1.0
    fit_intercept: bool = True
    normalize: bool = False
    lambda_sensitivity: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "values": []})

@dataclass
class DomainPriorConfig:
    min_intrinsic_influence: float = 0.01
    use_r_squared_intrinsic: bool = True
    fixed_intrinsic_influence: float = 0.3

@dataclass
class SamplingConfig:
    smoothing_alpha: float = 0.2
    min_sampling_prob: float = 0.001
    alpha_sensitivity: Dict[str, Any] = field(default_factory=lambda: {"enabled": False, "values": []})

@dataclass
class BaggingConfig:
    enabled: bool = True
    n_estimators: int = 100
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: Union[int, float, str] = "sqrt"
    n_jobs: int = -1
    class_weight: Optional[str] = None

@dataclass
class BoostingConfig:
    enabled: bool = True
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    max_features: Union[int, float, str] = "sqrt"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0
    reg_alpha: float = 0
    reg_lambda: float = 1
    n_jobs: int = -1
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    early_stopping_rounds: int = 10

@dataclass
class AlgorithmConfig:
    experiment_type: AlgorithmType = AlgorithmType.COMPARISON
    bagging: BaggingConfig = field(default_factory=BaggingConfig)
    boosting: BoostingConfig = field(default_factory=BoostingConfig)

@dataclass
class TemporalConfig:
    enabled: bool = True
    time_buckets: Dict[str, List[int]] = field(default_factory=lambda: {
        "morning": [6, 11],
        "noon": [11, 13],
        "afternoon": [14, 18]
    })

@dataclass
class EvaluationConfig:
    n_runs: int = 30
    random_seed_base: int = 42
    significance_level: float = 0.05
    correction_method: str = "bonferroni"
    mape_threshold: float = 0.05
    regression_metrics: List[str] = field(default_factory=lambda: ["MAE", "RMSE", "MAPE", "R2"])
    classification_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_roc"]
    )
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

@dataclass
class ExportConfig:
    enabled: bool = False
    format: str = "onnx"
    output_dir: str = "./models"
    onnx: Dict[str, Any] = field(default_factory=lambda: {"opset_version": 14, "target_platform": "mlnet"})

@dataclass
class OutputConfig:
    dir: str = "./dwrf_output"
    save_plots: bool = True
    save_models: bool = True
    save_predictions: bool = True
    save_config: bool = True
    save_sensitivity_results: bool = True
    save_temporal_results: bool = True

@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "dwrf_run.log"
    console_output: bool = True
    verbose_metrics: bool = True

@dataclass
class DWRFConfigV4:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    graphs: Dict[str, GraphConfig] = field(default_factory=dict)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    domain_prior: DomainPriorConfig = field(default_factory=DomainPriorConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: str) -> DWRFConfigV4:
    """Load YAML configuration and return DWRFConfigV4 object."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    def _build_visualization(viz_data):
        if not viz_data:
            return VisualizationConfig()
        colors = VisualizationColorsConfig(**viz_data.get('colors', {}))
        fonts = VisualizationFontsConfig(**viz_data.get('fonts', {}))
        styles = VisualizationStylesConfig(**viz_data.get('styles', {}))
        labels = VisualizationLabelsConfig(**viz_data.get('labels', {}))
        return VisualizationConfig(colors=colors, fonts=fonts, styles=styles, labels=labels)

    def _build_graphs(graphs_data):
        if not graphs_data:
            return {}
        return {name: GraphConfig(**g) for name, g in graphs_data.items()}

    def _build_evaluation(eval_data):
        if not eval_data:
            return EvaluationConfig()
        # Handle temporal separately
        temporal_data = eval_data.get('temporal', {})
        if isinstance(temporal_data, dict):
            temporal = TemporalConfig(**temporal_data)
        else:
            temporal = TemporalConfig()
        # Remove temporal from eval_data to avoid duplicate
        eval_copy = {k: v for k, v in eval_data.items() if k != 'temporal'}
        return EvaluationConfig(temporal=temporal, **eval_copy)

    def _build_algorithm(alg_data):
        if not alg_data:
            return AlgorithmConfig()
        # Handle bagging and boosting
        bagging_data = alg_data.get('bagging', {})
        boosting_data = alg_data.get('boosting', {})
        bagging = BaggingConfig(**bagging_data) if isinstance(bagging_data, dict) else BaggingConfig()
        boosting = BoostingConfig(**boosting_data) if isinstance(boosting_data, dict) else BoostingConfig()
        # Handle experiment_type
        exp_type = alg_data.get('experiment_type', 'comparison')
        if isinstance(exp_type, str):
            try:
                exp_type = AlgorithmType(exp_type)
            except ValueError:
                exp_type = AlgorithmType.COMPARISON
        return AlgorithmConfig(experiment_type=exp_type, bagging=bagging, boosting=boosting)

    # Build config
    cfg = DWRFConfigV4(
        experiment=ExperimentConfig(**data.get('experiment', {})),
        visualization=_build_visualization(data.get('visualization', {})),
        problem=ProblemConfig(**data.get('problem', {})),
        dataset=DatasetConfig(**data.get('dataset', {})),
        features=FeaturesConfig(**data.get('features', {})),
        graphs=_build_graphs(data.get('graphs', {})),
        ridge=RidgeConfig(**data.get('ridge', {})),
        domain_prior=DomainPriorConfig(**data.get('domain_prior', {})),
        sampling=SamplingConfig(**data.get('sampling', {})),
        algorithm=_build_algorithm(data.get('algorithm', {})),
        evaluation=_build_evaluation(data.get('evaluation', {})),
        export=ExportConfig(**data.get('export', {})),
        output=OutputConfig(**data.get('output', {})),
        logging=LoggingConfig(**data.get('logging', {}))
    )
    return cfg