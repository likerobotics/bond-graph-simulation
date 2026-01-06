
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


# Create model object
model = BondGraphModel(name='mass spring damper system with ground and external force')

# Create some elements (type is mandatory, other parameters are optional)
se = ElementFactory.create('SE')
r = ElementFactory.create('R')
i = ElementFactory.create('I')
c = ElementFactory.create('C')
sf = ElementFactory.create('SF')

j1_1 = ElementFactory.create('1')  # 1-junction for SE and I
j1_2 = ElementFactory.create('1')  # 1-junction for 0 - RС
j0_0 = ElementFactory.create('0')  # 0-junction between R and C
j1_3 = ElementFactory.create('1')  # 1-junction for SF


# Connect components properly using junctions
model.connect(se, j1_1)   # SE -> j1_1
model.connect(i, j1_1)    # I -> 1-junction 1

model.connect(r, j1_2)    # R -> 1-junction 2
model.connect(c, j1_2)    # C -> 1-junction 2

model.connect(j1_1, j0_0)    # 0junction -> 1-junction 1
model.connect(j0_0, j1_3)    # 0junction -> 1-junction 3
model.connect(j0_0, j1_2)    # 0junction -> 1-junction 2

model.connect(j1_3, sf)    # 1-junction 3 -> SF

# Apply causality rules
engine = RuleEngine(model, debug=False)
engine.apply_all()


custom_layered_layout(model) # make suitable layout
draw_bond_graph(model)# Draw the graph
model.debug_display_bonds() # display bonds info for debug  

