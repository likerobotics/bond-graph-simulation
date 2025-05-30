from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
import networkx as nx
from core.BondGraph2 import BondGraphModel
from core.base import ElementFactory

from dataclasses import dataclass, astuple, asdict, field

@dataclass(frozen=True, order=True)
class ErrorMsg:
    id: int
    text: str = field(default="")
    rules: list[int] = field(default_factory=list, compare=False, hash=False, repr=False)
        
class Status(Enum):
    ok = 1
    not_connected = 2
    not_assigned = 3

def BondGraphModel_from_nx(nx_graph: nx.Graph )-> BondGraphModel:
    """
    Creates a bondgraph model based on networkx graph.
    NB! It requires the extra configuration of bondgraph before simulation.
    """
    model = BondGraphModel("BG_generated_i_")

    # Сопоставление узлов NetworkX с элементами BondGraph
    node_to_element = {}
    for node_id, data in nx_graph.nodes(data=True):
        type_str = data.get("type", "R")  # тип по умолчанию: резистор
        name = data.get("name", f"{type_str}_{node_id}")
        position = data.get("position", [None, None])

        element = ElementFactory.create(type_str=type_str, name=name, position=position)
        model.elements.append(element)
        node_to_element[node_id] = element

    # Связываем элементы через порты
    for u, v in nx_graph.edges():
        from_elem = node_to_element[u]
        to_elem = node_to_element[v]
        model.connect(from_elem, to_elem)

    return model