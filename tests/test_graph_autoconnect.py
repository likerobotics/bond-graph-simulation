# test_graph_autoconnect.py
import unittest
from core.base import ElementFactory
from core.bondgraph import BondGraphModel


class TestGraphAutoConnect(unittest.TestCase):
    def test_auto_connection_sequence(self):
        # Create a sequence of elements
        model = BondGraphModel("AutoConnectTest")
        elements = [ElementFactory.create(t) for t in ['SE', 'R', 'C']]

        for el in elements:
            model.add_element(el)

        # Should be 3 elements, each with 2 ports
        self.assertEqual(len(model.elements), 3)
        self.assertEqual(len(model.ports), 6)

        # Should have 2 bonds (SE→R, R→C)
        self.assertEqual(len(model.bonds), 2)

        # Check that bonds connect correct ports
        self.assertEqual(model.bonds[0].from_port, elements[0].ports[-1])
        self.assertEqual(model.bonds[0].to_port, elements[1].ports[0])

        self.assertEqual(model.bonds[1].from_port, elements[1].ports[-1])
        self.assertEqual(model.bonds[1].to_port, elements[2].ports[0])


if __name__ == '__main__':
    unittest.main()
