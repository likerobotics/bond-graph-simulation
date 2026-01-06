
import sys
sys.path.append("../../")
from bond_graph_simulation.core.base import ElementFactory
from bond_graph_simulation.core.BondGraph2 import BondGraphModel
from bond_graph_simulation.rules.engine import RuleEngine
from bond_graph_simulation.equations.generator import EquationGenerator
from bond_graph_simulation.equations.statespace import CauchyFormGenerator
from bond_graph_simulation.equations.statespace import StateSpaceBuilder
from bond_graph_simulation.numerical.simulation import BondGraphSimulator


from bond_graph_simulation.inout.visualizer import draw_bond_graph, custom_layered_layout

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


model = BondGraphModel(name='GY model example')#  Create model

#Create some elements
# ZeroJ_1 = ElementFactory.create('0', name='0j_2',)
# SF_1 = ElementFactory.create('SF', name='SF_1', )

GY_1 = ElementFactory.create('GY')

SE_1 = ElementFactory.create('SE', name='SE_1', )
R_1 = ElementFactory.create('R', name='R_1', )
R_2 = ElementFactory.create('R', name='R_2', )
I_1 = ElementFactory.create('I', name='I_1',)
I_2 = ElementFactory.create('I', name='I_2',)

OneJ_1 = ElementFactory.create('1', name='1j_4')
OneJ_2 = ElementFactory.create('1', name='1j_5')

# add a connections between elements in model
model.connect(SE_1, OneJ_1)
model.connect(I_1, OneJ_1)
model.connect(R_1, OneJ_1)
model.connect(GY_1, OneJ_1)
model.connect(GY_1, OneJ_2)
model.connect(OneJ_2, I_2)

model.connect(R_2, OneJ_2)


engine = RuleEngine(model, debug=False)
engine.apply_all()

problems = engine.find_invalid_bonds()
if problems:
    print("Detected some unvalid bonds:")
    for p in problems:
        print(p)
else:
    print("Allright!")

# 5) Generate equations (NOT NESS later Cochy will do it)
eqgen = EquationGenerator(model, debug=True)
eqs = eqgen.generate_equations()
# eqgen.visualize_equations()
print("Equations:", eqs)


cform = CauchyFormGenerator(model, debug=False)
# cform.debug = True
eqs = cform.build_cauchy_form()
print("Cauchy form equations:", eqs)


print("User have to define the output variables for the system to make possive the output equations generation")
print(cform.get_all_ef_variables())

# cform.interactive_generate_output_equations()
# Instead of input we can provide variables via code
cform.generate_output_equations('e6,f6')

print(f"System has variables:{cform.final_vars}")


ssb = StateSpaceBuilder(model, cform, debug=False)
A, B, C, D = ssb.build_state_space()

print("A, B , C, D =\n", A, B , C, D)

sim = BondGraphSimulator(model, ssb)

sim.print_simulation_requirements()  # shows the required parameters (order is important)

