# test_graph_parser.py
import unittest
import networkx as nx
from io.parser import from_nx_graph
from core.base import ElementType


class TestGraphParser(unittest.TestCase):
    def test_parse_simple_graph(self):
        # Build a simple networkx graph: SE -> R -> C
        G = nx.DiGraph()
        G.add_node("SE", type="SE")
        G.add_node("R", type="R")
        G.add_node("C", type="C")

        G.add_edge("SE", "R")
        G.add_edge("R", "C")

        model = from_nx_graph(G, name="TestGraph")

        self.assertEqual(len(model.elements), 3)
        self.assertEqual(len(model.bonds), 2)

        names = [e.name for e in model.elements]
        self.assertIn("SE", names)
        self.assertIn("R", names)
        self.assertIn("C", names)

        types = [e.type for e in model.elements]
        self.assertIn(ElementType.SOURCE_EFFORT, types)
        self.assertIn(ElementType.RESISTOR, types)
        self.assertIn(ElementType.CAPACITOR, types)


if __name__ == '__main__':
    unittest.main()
