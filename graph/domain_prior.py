# graph/domain_prior.py
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DomainPriorCalculator:
    """
    Compute Domain Prior Weights (DPWs) via dynamic programming on a causal DAG.

    Influence is propagated backwards from the target. For each node that receives
    influence I:

      γ = intrinsic self-influence fraction
        - if use_r_squared_intrinsic: γ = max(min_intrinsic, 1 - R²)
        - else: γ = fixed_intrinsic_influence (clamped to [min_intrinsic, 1])

      - γ · I stays attributed to the node itself (intrinsic residual)
      - (1 - γ) · I is distributed to its parents according to the DRW weights

    This matches the SDS specification that intermediate nodes retain a residual
    self-influence based on unexplained variance (or a fixed value).
    """

    def __init__(self, min_intrinsic_influence: float = 0.01,
                 use_r_squared_intrinsic: bool = True,
                 fixed_intrinsic_influence: float = 0.3):
        self.min_intrinsic = float(min_intrinsic_influence)
        self.use_r_squared = bool(use_r_squared_intrinsic)
        self.fixed_intrinsic = float(fixed_intrinsic_influence)

    def _get_gamma(self, node: str, r2_scores: Optional[Dict[str, float]]) -> float:
        """Compute the intrinsic (self) influence fraction γ for a node."""
        if self.use_r_squared and r2_scores is not None and node in r2_scores:
            r2 = r2_scores[node]
            gamma = 1.0 - r2
        else:
            gamma = self.fixed_intrinsic
        # Clamp
        gamma = max(self.min_intrinsic, min(1.0, gamma))
        return gamma

    def compute(self, graph, drw: Dict[str, Dict[str, float]],
                target_node: str, input_features: List[str],
                column_mapping: Dict[str, str],
                r2_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute DPW as influence on target using original graph (cause → effect).
        Propagate backwards from target to its ancestors, retaining intrinsic
        self-influence at each node according to R² (or fixed value).

        Args:
            graph: ExpertGraph instance
            drw: dict node -> {parent: weight}
            target_node: name of the target column / node
            input_features: list of feature names that will be used by the model
            column_mapping: graph node name -> actual column name
            r2_scores: optional dict node -> R² from RidgeEstimator (for intrinsic)

        Returns:
            Dict of feature_name -> normalised DPW (sums to 1 over input_features)
        """
        reverse_map = {v: k for k, v in column_mapping.items()}
        target_graph_node = reverse_map.get(target_node, target_node)
        if target_graph_node not in graph.get_all_nodes():
            raise ValueError(f"Target node '{target_graph_node}' not in graph.")

        # Get topological order (causes → effects)
        topo = graph.get_topological_order()

        # Initialize influence for all nodes
        influence = {node: 0.0 for node in graph.get_all_nodes()}
        influence[target_graph_node] = 1.0   # target starts with full influence

        # Process nodes in reverse topological order (from effects → causes)
        # so that when we process a node, all of its descendants have already
        # contributed their residual influence to it.
        for node in reversed(topo):
            I = influence[node]
            if I <= 0:
                continue

            parents = graph.get_parents(node)
            if not parents:
                # Root node: all remaining influence stays with the node
                continue

            gamma = self._get_gamma(node, r2_scores)
            # Propagate only the non-intrinsic portion upstream.
            # The gamma * I remains attributed to this node.
            to_propagate = (1.0 - gamma) * I

            node_drw = drw.get(node, {})
            if node_drw:
                total_w = sum(node_drw.get(p, 0.0) for p in parents)
                if total_w > 0:
                    for parent in parents:
                        w = node_drw.get(parent, 0.0)
                        if w > 0:
                            influence[parent] += to_propagate * (w / total_w)
                else:
                    # Degenerate DRW – uniform
                    uniform = to_propagate / len(parents)
                    for parent in parents:
                        influence[parent] += uniform
            else:
                # No DRW available – distribute uniformly
                uniform = to_propagate / len(parents)
                for parent in parents:
                    influence[parent] += uniform

            # After propagation the node retains gamma * I as its intrinsic share
            # (influence[node] is left unchanged).

        # Extract DPW for input features only
        result = {}
        for feat in input_features:
            # Find graph node corresponding to this feature
            graph_node = None
            for node, col in column_mapping.items():
                if col == feat:
                    graph_node = node
                    break
            if graph_node is None:
                graph_node = feat
            result[feat] = influence.get(graph_node, 0.0)

        total = sum(result.values())
        if total == 0:
            logger.warning("No influence paths found; using uniform DPW.")
            n = len(result)
            for feat in result:
                result[feat] = 1.0 / n if n > 0 else 0.0
        else:
            for feat in result:
                result[feat] /= total

        logger.info(f"Final DPW for input features (with intrinsic): {result}")
        return result


def apply_sampling_smoothing(weights, alpha=0.2, min_prob=0.0):
    """Equation (21): P_j = (1-α) w_j + α/p, then optional floor and renormalise.

    Applied to already-normalised DPWs before feature sampling in DARF / DW-GB.
    Raw (unsmoothed) DPWs should still be stored for the domain-prior plot.
    """
    if not weights:
        return weights
    names = list(weights.keys())
    p = len(names)
    if p == 0:
        return weights
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        alpha = 0.0
    alpha = max(0.0, min(1.0, alpha))
    try:
        min_prob = float(min_prob)
    except (TypeError, ValueError):
        min_prob = 0.0

    smoothed = {f: (1.0 - alpha) * float(weights[f]) + alpha / p for f in names}
    if min_prob > 0:
        smoothed = {f: max(min_prob, w) for f, w in smoothed.items()}
    total = sum(smoothed.values())
    if total <= 0:
        return {f: 1.0 / p for f in names}
    return {f: w / total for f, w in smoothed.items()}
