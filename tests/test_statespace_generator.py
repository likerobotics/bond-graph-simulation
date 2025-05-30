# test_statespace_builder_v2.py
import unittest
import sympy as sp

from core.base import ElementFactory, ElementType
from core.bondgraph import BondGraphModel
from rules.engine import RuleEngine
from equations.statespace import StateSpaceBuilder

class TestStateSpaceBuilderV2(unittest.TestCase):
    def test_build_state_space_rich_model(self):
        """
        1) Create the standard RLC with junctions model.
        2) Apply causality.
        3) Use StateSpaceBuilder to build matrix form.
        4) Check A, B for correct dimensions.
        """
        # 1) Create model
        model = BondGraphModel("RLC with junctions")

        # Create elements
        se = ElementFactory.create('SE')
        r  = ElementFactory.create('R')
        i  = ElementFactory.create('I')
        c  = ElementFactory.create('C')
        sf = ElementFactory.create('SF')

        j1 = ElementFactory.create('1')
        j2 = ElementFactory.create('1')
        j0 = ElementFactory.create('0')
        j3 = ElementFactory.create('1')

        # Connect them
        model.connect(se, j1)
        model.connect(i, j1)
        model.connect(r, j2)
        model.connect(c, j2)
        model.connect(j0, j2)
        model.connect(j0, j3)
        model.connect(j3, sf)

        # 2) Apply causality
        engine = RuleEngine(model)
        engine.apply_all()

        # 3) Build matrix form with StateSpaceBuilder
        ssb = StateSpaceBuilder(model, debug=True)
        A, B = ssb.build_state_space()

        # 4) Check matrix dimensions
        # Count states: 1 capacitor (C) + 1 inductor (I) => 2 states
        # Count inputs: SE + SF => 2 inputs
        # A should be 2x2, B should be 2x2 if it's purely linear

        # We can find how many capacitors/inductors
        c_elems = [e for e in model.elements if e.type == ElementType.CAPACITOR]
        i_elems = [e for e in model.elements if e.type == ElementType.INDUCTOR]
        num_states = len(c_elems) + len(i_elems)

        # how many sources
        se_elems = [e for e in model.elements if e.type == ElementType.SOURCE_EFFORT]
        sf_elems = [e for e in model.elements if e.type == ElementType.SOURCE_FLOW]
        num_inputs = len(se_elems) + len(sf_elems)

        # A, B might be None if the system isn't purely linear, but let's assume for the test
        self.assertIsNotNone(A, "Matrix A is None.")
        self.assertIsNotNone(B, "Matrix B is None.")

        self.assertEqual(A.shape, (num_states, num_states),
                         "Matrix A dimension mismatch.")
        self.assertEqual(B.shape, (num_states, num_inputs),
                         "Matrix B dimension mismatch.")

        # Optionally, inspect the symbolic content
        print("A =", A)
        print("B =", B)

if __name__ == '__main__':
    unittest.main()