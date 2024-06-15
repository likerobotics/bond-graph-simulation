from enum import Enum
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import BondGraph as bg

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

def BondGpraph_from_model(G: nx.Graph )-> bg.BondGraph:
    """
    Creates a bondgraph model based on networkx graph.
    NB! It requires the extra configuration of bondgraph before simulation.
    """
    model =  bg.BondGraph(Name ='Bond graph from nx graph')
    model.reset()
    connections_list = []
    items_pursed = {}
    for i in G.edges:
        if i[0] not in list(items_pursed.keys()):
            item = bg.BGelement(i[0].split('-')[0])
            model.addElement(item)
            items_pursed[i[0]] = item
        if i[1] not in list(items_pursed.keys()):
            item = bg.BGelement(i[1].split('-')[0])
            model.addElement(item)
            items_pursed[i[1]] = item
        if i not in connections_list:
            model.connect(items_pursed[i[0]], items_pursed[i[1]])
            connections_list.append(i)
            
    return model