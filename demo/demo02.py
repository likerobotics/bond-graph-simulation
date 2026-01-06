
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
model = BondGraphModel(name='Serial 2 mass connection with 2 springs and one damper and ground connection + external force')

# Create some elements (type is mandatory, other parameters are optional)
SE_2 = ElementFactory.create('SE', name='SE_2', )
SE_3 = ElementFactory.create('SE', name='SE_3',)
SF_2 = ElementFactory.create('SF', name='SF_2', )

C_2 = ElementFactory.create('C', name='C_2', )
C_3 = ElementFactory.create('C', name='C_3', )
R_2 = ElementFactory.create('R', name='R_2', )
I_2 = ElementFactory.create('I', name='I_2',)
I_3 = ElementFactory.create('I', name='I_3', )

OneJ_4 = ElementFactory.create('1', name='1j_4')
OneJ_5 = ElementFactory.create('1', name='1j_5')
OneJ_6 = ElementFactory.create('1', name='1j_6', )
OneJ_7 = ElementFactory.create('1', name='1j_7', )
OneJ_8 = ElementFactory.create('1', name='1j_8', )

ZeroJ_2 = ElementFactory.create('0', name='0j_2',)
ZeroJ_3 = ElementFactory.create('0', name='0j_3',)

# Connect elements to build the bond graph
model.connect(SE_2, OneJ_4)
model.connect(SE_3, OneJ_5)

model.connect(I_2, OneJ_4)
model.connect(OneJ_4, ZeroJ_2)
model.connect(ZeroJ_2, OneJ_5)
model.connect(ZeroJ_2, OneJ_6)
model.connect(C_2, OneJ_6)
model.connect(R_2, OneJ_6)
model.connect(OneJ_5, I_3)
model.connect(OneJ_5, ZeroJ_3)
model.connect(OneJ_7, ZeroJ_3)
model.connect(C_3, OneJ_7)
model.connect(OneJ_8, ZeroJ_3)
model.connect(OneJ_8, SF_2)

# Apply causality rules
engine = RuleEngine(model, debug=False)
engine.apply_all()

