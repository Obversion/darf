# models/factory.py
from .bagging import DomainWeightedRandomForest
from .boosting import DomainWeightedGradientBoosting, DomainWeightedXGBoost


class ModelFactory:
    @staticmethod
    def create_model(config, algorithm_type, feature_weights=None,
                     num_classes=None, feature_names=None, **kwargs):
        """
          bagging  + weights=None → RF baseline (DomainWeightedRandomForest, uniform)
          bagging  + weights=DPW  → DWRF
          boosting + weights=None → XGBoost baseline
          boosting + weights=DPW  → DomainWeightedGradientBoosting
        """
        if algorithm_type == "bagging":
            params = {
                'n_estimators': config.algorithm.bagging.n_estimators,
                'max_depth': config.algorithm.bagging.max_depth,
                'min_samples_split': config.algorithm.bagging.min_samples_split,
                'min_samples_leaf': config.algorithm.bagging.min_samples_leaf,
                'max_features': config.algorithm.bagging.max_features,
                'n_jobs': config.algorithm.bagging.n_jobs,
                'class_weight': config.algorithm.bagging.class_weight,
            }
            params.update(kwargs)
            return DomainWeightedRandomForest(
                config,
                feature_weights=feature_weights,
                num_classes=num_classes,
                feature_names=feature_names,
                **params,
            )

        if algorithm_type == "boosting":
            boost_cfg = config.algorithm.boosting
            if feature_weights is not None:
                params = {
                    'n_estimators': boost_cfg.n_estimators,
                    'max_depth': boost_cfg.max_depth,
                    'learning_rate': boost_cfg.learning_rate,
                    'subsample': boost_cfg.subsample,
                    'max_features': getattr(boost_cfg, 'max_features', 'sqrt'),
                    'min_samples_split': getattr(boost_cfg, 'min_samples_split', 2),
                    'min_samples_leaf': getattr(boost_cfg, 'min_samples_leaf', 1),
                    'n_jobs': boost_cfg.n_jobs,
                }
                params.update(kwargs)
                return DomainWeightedGradientBoosting(
                    config,
                    feature_weights=feature_weights,
                    num_classes=num_classes,
                    feature_names=feature_names,
                    **params,
                )
            # XGBoost baseline
            params = {
                'n_estimators': boost_cfg.n_estimators,
                'max_depth': boost_cfg.max_depth,
                'learning_rate': boost_cfg.learning_rate,
                'subsample': boost_cfg.subsample,
                'colsample_bytree': boost_cfg.colsample_bytree,
                'min_child_weight': boost_cfg.min_child_weight,
                'gamma': boost_cfg.gamma,
                'reg_alpha': boost_cfg.reg_alpha,
                'reg_lambda': boost_cfg.reg_lambda,
                'n_jobs': boost_cfg.n_jobs,
                'objective': boost_cfg.objective,
                'eval_metric': boost_cfg.eval_metric,
                'early_stopping_rounds': boost_cfg.early_stopping_rounds,
            }
            params.update(kwargs)
            return DomainWeightedXGBoost(
                feature_weights=None,
                num_classes=num_classes,
                **params,
            )

        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
