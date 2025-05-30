# test_ruleengine_applyall.py
import unittest

import sys
sys.path.append("../")

from core.base import ElementFactory, ElementType
from core.bondgraph import BondGraphModel
from rules.engine import RuleEngine

class TestRuleEngineApplyAll(unittest.TestCase):
    def test_apply_all_rich_model(self):
        # Create the 'RLC with junctions' model (our standard example)
        model = BondGraphModel("RLC with junctions")

        # Create elements
        se = ElementFactory.create('SE')
        r = ElementFactory.create('R')
        i = ElementFactory.create('I')
        c = ElementFactory.create('C')
        sf = ElementFactory.create('SF')

        j1 = ElementFactory.create('1')  # 1-junction for SE and I
        j2 = ElementFactory.create('1')  # 1-junction for R, C, 0
        j0 = ElementFactory.create('0')  # 0-junction between R and C
        j3 = ElementFactory.create('1')  # 1-junction for SF

        # Connect them (based on the user's standard model)
        model.connect(se, j1)
        model.connect(i, j1)
        model.connect(r, j2)
        model.connect(c, j2)
        model.connect(j0, j2)
        model.connect(j0, j3)
        model.connect(j3, sf)

        # Apply rules
        engine = RuleEngine(model)
        engine.apply_all()

        # After apply_all, we expect all ports to have direction & causality set
        missing_any = False
        for element in model.elements:
            for port in element.ports:
                if port.direction is None or port.causality is None:
                    missing_any = True
                    print(f"[ERROR] Port in element {element.name} is incomplete: direction={port.direction}, causality={port.causality}")
        self.assertFalse(missing_any, "Some ports remain unassigned after apply_all")

        # Additionally, we can do basic checks, e.g. R should be Input+Uncausal
        # We'll just check the first port for demonstration
        for element in model.elements:
            if element.type == ElementType.RESISTOR:
                p = element.ports[0]
                self.assertEqual(p.direction, 'Input')
                self.assertEqual(p.causality, 'Uncausal')
            elif element.type == ElementType.CAPACITOR:
                p = element.ports[0]
                self.assertEqual(p.direction, 'Input')
                self.assertEqual(p.causality, 'Uncausal')
            elif element.type == ElementType.INDUCTOR:
                p = element.ports[0]
                self.assertEqual(p.direction, 'Input')
                self.assertEqual(p.causality, 'Causal')
            elif element.type == ElementType.SOURCE_EFFORT:
                p = element.ports[0]
                self.assertEqual(p.direction, 'Output')
                self.assertEqual(p.causality, 'Uncausal')
            elif element.type == ElementType.SOURCE_FLOW:
                p = element.ports[0]
                self.assertEqual(p.direction, 'Input')
                self.assertEqual(p.causality, 'Causal')
            # For junctions and TF/GY, the test can be more elaborate

if __name__ == '__main__':
    unittest.main()
