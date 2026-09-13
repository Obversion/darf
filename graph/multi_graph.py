# graph/multi_graph.py
from typing import Dict, List
import logging
from .expert_graph import ExpertGraph
from config.config import DWRFConfigV4

logger = logging.getLogger(__name__)

class MultiGraphManager:
    def __init__(self, config: DWRFConfigV4):
        self.config = config
        self.graphs = {}  # Original graphs (cause -> effect)

    def build_all_graphs(self) -> Dict[str, ExpertGraph]:
        """Build all configured graphs (original direction: cause -> effect)."""
        for graph_name, graph_cfg in self.config.graphs.items():
            edges = graph_cfg.edges
            self.graphs[graph_name] = ExpertGraph(edges)
            logger.info(f"Graph {graph_name} edges: {edges}")
        return self.graphs

    def get_graph_names(self) -> List[str]:
        return list(self.graphs.keys())