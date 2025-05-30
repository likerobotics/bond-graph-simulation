# test_equation_generator.py
import unittest
import sympy
from core.bondgraph import BondGraphModel
from core.base import ElementFactory, ElementType
from rules.engine import RuleEngine
from equations.generator import EquationGenerator

class TestEquationGenerator(unittest.TestCase):
    def test_generate_equations_on_rich_model(self):
        # 1) Build the 'RLC with junctions' model
        model = BondGraphModel("RLC with junctions")

        # 2) Create elements
        se = ElementFactory.create('SE')
        r  = ElementFactory.create('R')
        i  = ElementFactory.create('I')
        c  = ElementFactory.create('C')
        sf = ElementFactory.create('SF')

        j1 = ElementFactory.create('1')  # 1-junction for SE and I
        j2 = ElementFactory.create('1')  # 1-junction for R, C, 0
        j0 = ElementFactory.create('0')  # 0-junction between R and C
        j3 = ElementFactory.create('1')  # 1-junction for SF

        # 3) Connect
        model.connect(se, j1)
        model.connect(i, j1)
        model.connect(r, j2)
        model.connect(c, j2)
        model.connect(j0, j2)
        model.connect(j0, j3)
        model.connect(j3, sf)

        # 4) Apply rules (direction + causality)
        engine = RuleEngine(model)
        engine.apply_all()

        # 5) Generate equations
        eqgen = EquationGenerator(model, debug=True)
        eqs = eqgen.generate_equations()

        self.assertGreater(len(eqs), 0, "No equations generated.")

        # 6) Some partial checks:
        #    We'll look for symbolic equations for R, C, I, SE, SF, 0/1 junctions.
        #    We won't do a full string compare, but ensure some key forms appear.
        found_res = False
        found_cap = False
        found_ind = False
        found_se  = False
        found_sf  = False

        for eq in eqs:
            # eq is a Sympy Eq, eq.lhs and eq.rhs are expressions.
            expr_text = str(eq)

            if 'R' in expr_text and 'f' in expr_text and 'e' in expr_text:
                found_res = True
            if '/C' in expr_text and 'q' in expr_text and 'e' in expr_text:
                found_cap = True
            if '/I' in expr_text and 'p' in expr_text and 'f' in expr_text:
                found_ind = True
            if 'SE' in expr_text and 'e' in expr_text:
                found_se = True
            if 'SF' in expr_text and 'f' in expr_text:
                found_sf = True

        self.assertTrue(found_res, "Did not find resistor equation (R*f).")
        self.assertTrue(found_cap, "Did not find capacitor equation (q/C).")
        self.assertTrue(found_ind, "Did not find inductor equation (p/I).")
        self.assertTrue(found_se,  "Did not find source effort eq (SE).")
        self.assertTrue(found_sf,  "Did not find source flow eq (SF).")

        # Also check that 0/1 junction eqs are generated (common e, sum f = 0, or sum e = 0, common f)
        # Just a partial check:
        found_0junction = any('e' in str(eq.lhs) and str(eq.rhs) == '0' for eq in eqs) or \
                          any('f' in str(eq.lhs) and str(eq.rhs) == '0' for eq in eqs)
        found_1junction = any('e' in str(eq.rhs) and str(eq.lhs) == '0' for eq in eqs) or \
                          any('f' in str(eq.rhs) and str(eq.lhs) == '0' for eq in eqs)

        # It's not guaranteed this exact pattern, but at least we check if there's eq with 'sum(...)=0'
        # More robust check would parse the eq properly.

        self.assertTrue(found_0junction or found_1junction,
                        "No sign of 0/1-junction equations.")


if __name__ == '__main__':
    unittest.main()