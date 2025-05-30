# io/parser.py
import networkx as nx
from core.base import ElementFactory
from core.bondgraph import BondGraphModel


def from_nx_graph(nx_graph: nx.DiGraph, name="FromNetworkX") -> BondGraphModel:
    """
    Convert a generic networkx.DiGraph into a BondGraphModel.
    Nodes must have 'type' attributes ('R', 'C', etc.)
    Edges define connection direction between elements.
    """
    model = BondGraphModel(name)
    node_map = {}

    # Create and register elements
    for node_id, data in nx_graph.nodes(data=True):
        type_str = data.get("type")
        if not type_str:
            raise ValueError(f"Node {node_id} missing 'type' attribute")
        element = ElementFactory.create(type_str, name=str(node_id))
        node_map[node_id] = element

    # Connect elements using edges
    for u, v in nx_graph.edges():
        src = node_map[u]
        tgt = node_map[v]
        model.connect(src, tgt)

    return model