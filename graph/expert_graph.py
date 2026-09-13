import networkx as nx
from typing import Dict, List, Set, Optional

class ExpertGraph:
    """DAG representing expert knowledge graph."""

    def __init__(self, edges: Dict[str, List[str]]):
        self.edges = edges
        self.graph = nx.DiGraph()
        self._build_graph()
        self._validate_dag()

    def _build_graph(self):
        for parent, children in self.edges.items():
            for child in children:
                self.graph.add_edge(parent, child)

    def _validate_dag(self):
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph contains cycles; must be a DAG.")

    def get_parents(self, node: str) -> List[str]:
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))

    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def get_topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.graph))

    def get_ancestors(self, node: str) -> Set[str]:
        return set(nx.ancestors(self.graph, node))

    def get_descendants(self, node: str) -> Set[str]:
        return set(nx.descendants(self.graph, node))

    def is_root(self, node: str) -> bool:
        return self.graph.in_degree(node) == 0

    def is_leaf(self, node: str) -> bool:
        return self.graph.out_degree(node) == 0