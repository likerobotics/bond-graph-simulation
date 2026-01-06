
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

# Create model
model = BondGraphModel(name='TF example')

#Create some elements
TF_1 = ElementFactory.create('TF', name='TF_1', )
SE_1 = ElementFactory.create('SE', name='SE_1', )

C_1 = ElementFactory.create('C', name='C_1', )
I_1 = ElementFactory.create('I', name='I_1',)

OneJ_1 = ElementFactory.create('1', name='1j_4')
OneJ_2 = ElementFactory.create('1', name='1j_5')
OneJ_3 = ElementFactory.create('1', name='1j_6', )
OneJ_4 = ElementFactory.create('1', name='1j_7', )

ZeroJ_1 = ElementFactory.create('0', name='0j_2',)

SF_1 = ElementFactory.create('SF', name='SF_1', )


# Connect elements
model.connect(SE_1, OneJ_1)
model.connect(I_1, OneJ_1)
model.connect(TF_1, OneJ_1)
model.connect(TF_1, OneJ_2)
model.connect(OneJ_2, ZeroJ_1)
model.connect(ZeroJ_1, OneJ_3)
model.connect(ZeroJ_1, OneJ_4)
model.connect(C_1, OneJ_3)
model.connect(OneJ_4, SF_1)

# Apply causality rules
engine = RuleEngine(model, debug=False)
engine.apply_all()

