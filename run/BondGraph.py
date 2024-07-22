# -*- coding: utf-8 -*-
"""
+------------------------------------------------------+
|(c) 2022-2024 @likerobotics                           |
|    Islam Bzhikhatlov - ITMO University,              |
|    Faculty of control systems and Robotics           |
+------------------------------------------------------+

"""
""" 
A Python Package for robots simulation on low-level
entities and functionalities based on BondGraph simulation in 1-D space.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import sympy as sp
import re

import networkx as nx
#---------------CORE PART------------------------>>>#

class BGelement( object ):
    id_generator = itertools.count(0) # first generated is 0
    def __init__(self, Type:str, Name = None, Position = [None, None]):
        """
        BGcomponent is a node in graph structure.
        """
        self.__id = next(self.id_generator)
        self.id = self.__id
        self.__type = Type
        if Name is None:
            self.__name = str(Type) + '_' + str(self.id)
        else:
            self.__name = Name
        self.__position = Position #list of coodinates, position just for drawing
        self.value = 0 # the parameter of element, for example, stiffness for capacitor
        self.icon = ''
        self.energyDomain = None
        self.ports = [] # List: associated port objects
        self.input = 0
        self.output = 0
        self.effort = None
        self.flow = None
        self.stateVariable = None
        self.inputVariable = None
        self.parameter = None
        self.stateEquation = None
        self.outputEquation = None
        self.common = None
        
        # process, if the type is capacitor -> set the equations
        if Type == 'I':
            self.icon = 'I'
            self.parameter = 'I' + str(self.__id)
            self.stateVariable = 'p' + str(self.__id)
        if Type == 'C':
            self.icon = 'C'
            self.parameter = 'C' + str(self.__id)
            self.stateVariable = 'q' + str(self.__id)
        if Type == 'R':
            self.icon = 'R'
            self.parameter = '+R' + str(self.__id)
        if Type == 'SE':
            self.effort = '+SE' + str(self.__id)
            self.icon = 'SE'
            self.inputVariable = 'SE' + str(self.__id)
        if Type == 'SF':
            self.flow = '+SF' + str(self.__id)
            self.icon = 'SF'
            self.inputVariable = 'SF' + str(self.__id)
        if Type == '1':
            self.icon = '1'
        if Type == '0':
            self.icon = '0'
        if Type == 'TF':
            self.icon = 'TF'
            self.parameter = 'n' + str(self.__id)
        if Type == 'GY':
            self.icon = 'GY'
            self.parameter = 'm' + str(self.__id)
    def __str__(self):
        display =  'BGelement::_____:___________________\n'
        display += '          :   id: %d\n' % self.__id
        display += '          : name: %s\n' % self.__name
        display += '          : Type: %s\n' % self.__type
        display += '          :  pos: %s\n' % self.__position
        display += '          :ports: %s\n' % self.ports
        return display
    
    def setType(self, Variable = None):
        self.__type = Variable
        
    def getType(self):
        return self.__type
        
    def setName(self, Variable = None):
        self.__name = Variable
        
    def getName(self):
        return self.__name
        
    def setPosition(self, Variable = None):
        self.__position = Variable
        
    def getPosition(self):
        return self.__position

    def getId(self):
        return self.__id
    
    def addElement(self, Name = None, Type = None, Position = [0,0]):
        self.add_node(Name, position = Position)
        
    def setStateEquation(self, Equation = None):
        self.stateEquation = Equation
        
    def getStateEquation(self):
        return self.stateEquation
        
    def setOutputEquation(self, Equation = None):
        self.outputEquation = Equation
        
    def getOutputEquation(self):
        return self.outputEquation
        
    def setstateVariable(self, Variable = None):
        self.stateVariable = Variable
        
    def getstateVariable(self):
        return self.stateVariable
    
    def setinputVariable(self, Variable = None):
        self.inputVariable = Variable
        
    def getinputVariable(self):
        return self.inputVariable
    
    def setParam(self, Variable = None):
        self.parameter = Variable
        
    def getParam(self):
        return self.parameter
    
    def setValue(self, Value = None):
        self.value = Value
        return self.value
    
    def getValue(self):
        return self.value
    
    def getFlow(self):
        return self.flow
    
    def setFlow(self, Flow = None):
        self.flow = Flow
        return self.flow
    
    def getEffort(self):
        return self.effort
    
    def setEffort(self, Flow = None):
        self.effort = Flow
        return self.effort
    
    def setPorts(self, Ports):
        '''NOT RECOMENDED TO USE ! USE addPort'''
        self.ports = Ports
        return self.ports
    
    def addPort(self, port):
        self.ports.append(port)
        return self.ports
    
    def delPorts(self, port):
        self.ports.remove(port)
        return self.__ports
    
    def getPorts(self):
        return self.ports

    def getType(self):
        return self.__type

#<<<-------------------------BGport---------------------------->>>#
class BGport( object ):
    id_generator = itertools.count(0) # first generated is 0
    __ID = 0
    
    def __init__(self, Name = None, Type = None, Direction = None, Causality = None):
        """ 
        Ports is required to make possible the connection of several Bonds to Elements (also to distinct the input and outputs, and related data)
        The ports are created automatically when we call "Connect" two nodes(elements), it is not recommended to work with ports manually.
        """
        self.id = next(self.id_generator)
        self.__id = self.id
        self.__type = Type #power or signal
        self.__name = Name
        self.__position = 0 #position just for drawing
        self.__value = 0
        self.__direction = Direction # Input or Output
        self.__causality = Causality # Causal or Uncausal
        self.effort = None
        self.flow = None
        
    def __str__(self):
        display =  'BGport::_____:___________________\n'
        display += '          :   id: %d\n' % self.__id
        display += '          : Name: %s\n' % self.__name
        display += '          : Type: %s\n' % self.__type
        display += '          :  Direction: %s\n' % self.__direction
        display += '          :  Causality: %s\n' % self.__causality
        return display
        
    def setType(self, Variable = None):
        self.__type = Variable
        
    def getType(self):
        return self.__type
        
    def setName(self, Variable = None):
        self.__name = Variable
        
    def getName(self):
        return self.__name
        
    def setPosition(self, Variable = None):
        self.__position = Variable
    
    def getPosition(self):
        return self.__position

    def getId(self):
        return self.__id
    
    def setDirection(self, Variable = None):
        self.__direction = Variable
        
    def getDirection(self):
        return self.__direction
    
    def setCausality(self, Variable = None):
        self.__causality = Variable
        
    def getCausality(self):
        return self.__causality
       
class BGbond( object ):
    """
    The connection between nodes(bondes) through the ports
    Bond can not exist without assigned Ports.
    
    """
    id_generator = itertools.count(0) # first generated is 0 1
    __ID = 0
    
    def __init__(self, fromPort:BGport, toPort:BGport, Type = 'PowerBond'):
#         """ initializes a BGcomponent object """
        self.id = next(self.id_generator)
        self.__id = self.id
        self.__fromPort = fromPort # object
        self.__toPort = toPort #object
        self.effort = ''
        self.flow = ''
        self.__causalityStroke = None #if effort going out from left side item we use 1 else 0 ! #TODO use ports directly may be 
        self.__directionArrow = None #if point(positive energy) out from node 1 else 0 !!!!!!!!!!!!!!!!deprecated
        self.__type = Type
        
    def __str__(self):
        display =  'BGbond::_____:___________________\n'
        display += '       :   id: %d\n' % self.__id
        display += '       : from: %s\n' % self.__fromPort
        display += '       :   to: %s\n' % self.__toPort
        display += '       : Type: %s\n'   % self.__type
        display += '       : effort: %s\n' % self.effort
        display += '       :   flow: %s\n' % self.flow
        return display

    def setType(self, Variable = None):
        self.__type = Variable
        
    def getType(self):
        return self.__type
        
    def setFromPort(self, port:BGport):
        self.__fromPort = port
        
    def getFromPort(self):
        return self.__fromPort
        
    def setToPort(self, port:BGport):
        self.__toPort = port
        
    def getToPort(self):
        return self.__toPort

    def getId(self):
        return self.__id
        
    def setCausalityStroke(self, Variable = None):
        self.__causalityStroke = Variable
        
    def getCausalityStroke(self):
        return self.__causalityStroke
        
    def setDirectionArrow(self, Variable = None):
        self.__directionArrow = Variable
        
    def getDirectionArrow(self):
        return self.__directionArrow
        
 #<<<----------------------------------------------------->>>#\

class BondGraph():
    """
    The main class representing a model of system or subsystem
    
    """
    id_generator = itertools.count(0) # first generated is 0
    __name = None
    debug = False
    def __init__(self, BondsList = [], ElementsList = [], PortsList = [], Name = None):
        """ initializes a Graph object """
        self.__id = next(self.id_generator)
        self.__bondsList = BondsList #connections between node
        self.__elementsList = ElementsList #nodes
        self.__portsList = PortsList # ports
        self.__name = Name
        self.equastions = None
        self.equastions_sp = [] # simpy equation
        self.all_variables = [] # including efforts and flows
        self.final_variables = [] # except effort and flows
        self.state_variables = [] # p,q
        self.output_variables = [] # e1,f1
        self.input_variables = [] # SE1,SF2
        self.parameter_variables = [] #[C2, R3, I4] ex parameters
        self.capacitor_variables = [] # C2, I4
        self.eff_flows = [] # e1,e,4,f5,f8...
        self.__A = None
        self.__B = None
        self.__C = None
        self.__D = None

    def __str__(self):
        display =  'BondGraph::_____:___________________\n'
        display += '                : Name: %s\n' % self.__name
        display += '                : elementsList: %s\n' % self.__elementsList
        display += '                : bondsList: %s\n' % self.__bondsList
        display += '                : portsList: %s\n' % self.__portsList
        display += 'BG:BGelement::__:___________________\n'

    def get_matrix_A(self):
        """ Return A matrix """
        return self.__A
    
    def get_matrix_B(self):
        """ Return B matrix """
        return self.__B
    
    def get_matrix_C(self):
        """ Return C matrix """
        return self.__A
    
    def get_matrix_D(self):
        """ Return D matrix """
        return self.__B
    
    def reset(self):
        "clean all data in model"
        self.__elementsList = []
        self.__portsList = []
        self.__bondsList = []
    
    @property
    def id(self):
        return self.__id
    
    def check(self):
        print(f'This functionality carrently under development and will be updated soon!')
        #TODO  Check if any port, or bond or element is not connected
        # check if there are sympols that is not variables of system
        pass

    def addBond(self, Bond):
        """ add edge(connection between Bonds)"""
        self.__bondsList.append(Bond)
        
    def addElement(self, BGelement):
        """actually add the node, not edge"""
        self.__elementsList.append(BGelement)
        
    def getBondList(self):
        return self.__bondsList
    
    def getElementsList(self):
        return self.__elementsList
    
    def getPortsList(self):
        return self.__portsList
    
    def getStateEq(self):
        eq = []
        
        return eq

    #useful functions
    def adjacency_dict(self):
        """
        Returns the adjacency list representation of graph
        ! does not care about directions
        ! used only for drawing == position calculation
        """
        nodes = [element.id for element in self.__elementsList]
        adj = {node: [] for node in nodes}
        for bond in self.__bondsList:
            for element in self.__elementsList:
                if bond.getFromPort() in element.getPorts():
                    element_from = element.id
                if bond.getToPort() in element.getPorts():
                    element_to = element.id
            # append to adj_matrix
            adj[element_from].append(element_to)
            adj[element_to].append(element_from)
        return adj
    
    def render(self):
        '''
        reset positions for all bonds depended on connections
        
        RULE
        use adjacency dict to find node with max connections, others are placed around according the adjactecy
        '''
        
        #temp solution, could be executed only one time
        # RULE: find node with max edges
        adj = self.adjacency_dict()
        
        adj_dic_sorted = sorted(adj, key=lambda k: len(adj[k]), reverse = True)
        defined = []
        coords = {}
        pointer_x = 0
        pointer_y = 0
        for key in adj_dic_sorted:
            # 5
            if key in defined:
                pointer_x += 1
                pointer_y = 0
                for idx, item in enumerate(adj[key]):
                    if item not in defined:
                        coords[item] = [pointer_x, pointer_y]
                        defined.append(item)
                        pointer_y += 1
                    else:
                        pass
            else:
                coords[key] = [pointer_x, pointer_y]
                defined.append(key)
                pointer_x += 1
                # now check the connections
                for idx, item in enumerate(adj[key]):
                    if item in defined:
                        pass
                    else:
                        coords[item] = [pointer_x, pointer_y]
                        defined.append(item)
                        pointer_y += 1
        
        for i in self.__elementsList:
            pos_xy = coords[i.getId()]
            i.setPosition([pos_xy[0], pos_xy[1]])
        
    
    def connect(self, first_element:BGelement, second_element:BGelement):
        ''''
        Automatically create a bond between input Elements (only 2 elements allowed)
        one = BGelement
        second = BGelement
        '''
        # is this elements in the list of models elements?
        if first_element not in self.__elementsList:
            print(f'New BG element detected with ID={first_element.getId()}, adding to the model.')
            self.addElement(first_element)
        if second_element not in self.__elementsList:
            print(f'New BG element detected with ID={second_element.getId()}, adding to the model.')
            self.addElement(second_element)

        # Check is the nodes connected already
        for bond in self.getBondList():
            if bond.getFromPort() in first_element.getPorts() and bond.getToPort() in second_element.getPorts():
                print(f'Already connected... Sorry')
            if bond.getFromPort() in second_element.getPorts() and bond.getToPort() in first_element.getPorts():
                print(f'Already connected... Sorry')

        # create two ports for each elements
        head = BGport()
        tail = BGport()

        first_element.addPort(head)
        second_element.addPort(tail)
        
        # add new ports to model ports lis
        self.__portsList.append(head)
        self.__portsList.append(tail)
        # create bond and associate with ports
        Bond = BGbond(head, tail)
        Bond.setToPort(head)
        Bond.setFromPort(tail)
        self.__bondsList.append(Bond)
        
        return self
    
    def draw(self):
        """
        Dirty work, but useful just for visualization
        
        ____
        unrealedges == energy flow direction
        realedges  == effor direction (causality)
        
        """
        
        nodes = []
        for node in self.getElementsList():
            nodes.append(node.id)
        
        #Prepare data for drawing
        edge_num = []
        real_edges = [] # power flow direction
        unreal_edges = [] # causality direction
        bond_from = None
        bond_to = None

        port_pairs = []
        for bond in self.getBondList():
            port_pairs.append([bond.getFromPort(), bond.getToPort()])
        
        for port_pair in port_pairs:
            for element in self.getElementsList():
                if port_pair[0] in element.getPorts():
                    bond_from = element.id
                if port_pair[1] in element.getPorts():
                        bond_to = element.id
            edge_num.append(len(edge_num) + 1)
            real_edges.append((bond_from, bond_to))

            # unreal edges e.i effort directions for drowing (arrow side==causal stroke side)
            bond_from = None
            bond_to = None
            for element in self.getElementsList():
                if port_pair[0] in element.getPorts():
                    if port_pair[0].getCausality() == 'Uncausal':
                        bond_to = element.id
                    elif port_pair[0].getCausality() == 'Causal':
                        bond_from = element.id
                if port_pair[1] in element.getPorts():
                    if port_pair[1].getCausality() == 'Causal':
                        bond_from = element.id
                    elif port_pair[1].getCausality() == 'Uncausal':
                        bond_to = element.id
            if bond_from != None and bond_to != None:
                unreal_edges.append((bond_from, bond_to))
            else:
                print('WARNING 005: Some causality lost in drawings...')

        edge_labels = {}
        for i, ed in enumerate(real_edges):
            edge_labels[ed] = edge_num[i] - 1
        pos = {}
        G = nx.MultiDiGraph() # create object
        G.add_nodes_from(nodes)
        # nx.spring_layout(bgs)
        G.add_edges_from(real_edges)

        for node in self.getElementsList():
            pos[node.id] = node.getPosition()
        labels = {}
        for node in self.getElementsList():
            labels[node.id] = str(node.icon)
        
        #lets draw
        plt.figure(figsize=(19,8))
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:red", node_size = 700)
        nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color='b', arrowsize=16, arrows=True, connectionstyle='arc3, rad = 0.2')
        nx.draw_networkx_edges(G, pos, edgelist=unreal_edges, edge_color='g', arrows=True, connectionstyle='angle3, angleA=90, angleB=0', arrowstyle=']-, widthA=1.5, lengthA=0.2', min_source_margin=20, min_target_margin=20)
        nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color="whitesmoke", font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)        
        plt.margins(0.2)
        
    def display_PortsStatus(self):
        ListOfPorts = self.getPortsList()
        port_status = []
        for i in ListOfPorts:
            if i.getDirection() is None:
                port_status.append(None)
            else:
                port_status.append(i.getDirection())
        return port_status

    def display_Causality(self):
        ListOfPorts = self.getPortsList()
        port_status = []
        for i in ListOfPorts:
            if i.getCausality() is None:
                port_status.append(None)
            else:
                port_status.append(i.getCausality())
        return port_status
    
    def assign_sources_ports(self):
        """
        Automatically assign the energy flow direction and causality for POWER SOURCES.
        There is no garanty for correectness! Trying to make prefered forms.
        This method uses several rules that is correct for some implisit/explisit cases.
        Rules:
        SE is not causal, power goes out
        SF is causal, power goes in
        
        """
        for element in self.getElementsList():
            if element.getType() == 'SE':
                if self.debug: print('SE ports=', element.getPorts())
                for port in element.getPorts():
                    if port.getDirection() == None:
                        for bond in self.getBondList():
                            if bond.getFromPort() == port:
                                port.setDirection('Output') # asign port power flow type
                                port.setCausality('Uncausal')
                            elif bond.getToPort() == port:
                                if self.debug: print('Source of effort cant have power input, Redefining..')
                                tempo = bond.getToPort()
                                bond.setToPort(bond.getFromPort())
                                bond.setFromPort(tempo)
                                port.setDirection('Output') # power flow
                                port.setCausality('Uncausal')

            if element.getType() == 'SF':
                if self.debug: print('SF ports=', element.getPorts())
                for port in element.getPorts():
                    if port.getDirection() == None:
                        for bond in self.getBondList():
                            if bond.getFromPort() == port:
                                if self.debug: print('Source of flow cant have power output, Redefining..')
                                tempo = bond.getToPort()
                                bond.setToPort(bond.getFromPort())
                                bond.setFromPort(tempo)
                                port.setDirection('Input') # power flow
                                port.setCausality('Causal')
                            elif bond.getToPort() == port:
                                port.setDirection('Input') # power flow
                                port.setCausality('Causal')

    def assign_ports_for_CRI_elemts(self):
        ## care about C, R and I elements(nodes)
        for element in self.getElementsList():
            # C-RULE: power go in or out(reversable energy storage), CAUSALITY Effort only GO OUT(integral form)
            if element.getType() == 'C':
                if self.debug: print('C ports=', element.getPorts())
                for port in element.getPorts():
                    if port.getDirection() == None:
                        for bond in self.getBondList():
                            if bond.getFromPort() == port:
                                if self.debug: print('Capacitor cant have power input, Redefining..')
                                tempo = bond.getToPort()
                                bond.setToPort(bond.getFromPort())
                                bond.setFromPort(tempo)
                                port.setDirection('Input')
                                port.setCausality('Uncausal') # Rule
                            elif bond.getToPort() == port:
                                port.setDirection('Input')
                                port.setCausality('Uncausal') # Rule
            if element.getType() == 'R':
                if self.debug: print('R ports=', element.getPorts())
                # R-RULE: power go IN anyway(dissipative element), CAUSALITY Effort only GO OUT (to be linear form)
                for port in element.getPorts():
                    if port.getDirection() == None:
                        for bond in self.getBondList():
                            if bond.getFromPort() == port:
                                if self.debug: print('Dissipative elemnt cant have power flow out, Redefining..')
                                tempo = bond.getToPort()
                                bond.setToPort(bond.getFromPort())
                                bond.setFromPort(tempo)
                                port.setDirection('Input')
                                port.setCausality('Uncausal') # Rule
                            elif bond.getToPort() == port:
                                port.setDirection('Input')
                                port.setCausality('Uncausal') # Rule
            # I-RULE: power go in or out(reversable energy storage), CAUSALITY Effort only GO IN(integral form)
            if element.getType() == 'I':
                if self.debug: print('I ports=', element.getPorts())
                for port in element.getPorts():
                    if port.getDirection() == None:
                        for bond in self.getBondList():
                            if bond.getFromPort() == port:
                                if self.debug: print('I-element cant have power go out in linear form')
                                tempo = bond.getToPort()
                                bond.setToPort(bond.getFromPort())
                                bond.setFromPort(tempo)
                                port.setDirection('Input')
                                port.setCausality('Causal')
                            elif bond.getToPort() == port:
                                port.setDirection('Input')
                                port.setCausality('Causal')
    
    def assign_ports_TF_GY(self):
        # RULE: (only 2 ports must be connected) - power go in from one side and out on another side (power conservative element), 
        # CAUSALITY only for one port, causality can be reversed during auto asignment
        
        list_of_tf = [element for element in self.getElementsList() if element.getType() == 'TF']
        for element in list_of_tf:
            if self.debug: print('TF ports=', element.getPorts())
            #if one port is assigned port has no direction or causality
            for port in element.getPorts():
                connected_bonds = [bond for bond in self.getBondList() if bond.getToPort() == port or bond.getFromPort() == port]
                if port.getDirection() == None:
                    if self.debug: print('TF-element can have 1 input and 1 output only')
                    if connected_bonds[0].getToPort() == port:
                        # from TF to element
                        tempo = connected_bonds[0].getToPort()
                        connected_bonds[0].setToPort(connected_bonds[0].getFromPort())
                        connected_bonds[0].setFromPort(tempo)
                        port.setDirection('Input')
                        # if one input,  another is output also
                        for port2 in element.getPorts():
                            if port2 != port: port2.setDirection('Output')
                    elif connected_bonds[0].getFromPort() == port:
                        # from element to TF
                        port.setDirection('Output')
                        for port2 in element.getPorts():
                            if port2 != port: port2.setDirection('Input')
                elif port.getCausality() == None:
                    if self.debug: print('TF has NOne causality')
                    if connected_bonds[0].getToPort() == port:
                        #assing based on other side of bonds port causality
                        if connected_bonds[0].getToPort().getCausality() == 'Causal':
                            port.setCausality('Uncausal')
                            for port2 in element.getPorts():
                                if port2 != port: port2.setCausality('Causal')
                        elif connected_bonds[0].getFromPort().getCausality() == 'Uncausal':
                            port.setCausality('Causal')
                            for port2 in element.getPorts():
                                if port2 != port: port2.setCausality('Uncausal')
                        # assigning based on rule of TF, if one port is causal, another mast be uncausal and vice versa
                        alter_ports = [an_port for an_port in element.getPorts() if an_port != port and an_port.getCausality() != None]
                        if len(alter_ports) > 0:
                            if alter_ports[0].getCausality() == 'Causal':
                                port.setCausality('Uncausal')
                            elif alter_ports[0].getCausality() == 'Uncausal':
                                port.setCausality('Causal')
                            else:
                                print('Errror in conditions')
                else:
                    pass
                    # print('no info for assigning, skipping for TF')

        list_of_gy = [element for element in self.getElementsList() if element.getType() == 'GY']
        for element in list_of_gy:
            if self.debug: print('GY ports=', element.getPorts())
            #if one port is assigned port has no direction or causality
            for port in element.getPorts():
                connected_bonds = [bond for bond in self.getBondList() if bond.getToPort() == port or bond.getFromPort() == port]
                if port.getDirection() == None:                    
                    if self.debug: print('GY-element can have 1 input and 1 output only')
                    if connected_bonds[0].getToPort() == port:
                        # from GY to element
                        tempo = connected_bonds[0].getToPort()
                        connected_bonds[0].setToPort(connected_bonds[0].getFromPort())
                        connected_bonds[0].setFromPort(tempo)
                        port.setDirection('Input')
                        for port2 in element.getPorts():
                            if port2 != port: port2.setDirection('Output')
                    elif connected_bonds[0].getFromPort() == port:
                        # from element to GY
                        port.setDirection('Output')
                        for port2 in element.getPorts():
                            if port2 != port: port2.setDirection('Input')
                elif port.getCausality() == None:
                    if connected_bonds[0].getToPort() == port:
                        if connected_bonds[0].getFromPort().getCausality() == 'Causal':
                            port.setCausality('Uncausal')
                            for port2 in element.getPorts():
                                if port2 != port: port2.setCausality('Causal')
                        elif connected_bonds[0].getFromPort().getCausality() == 'Uncausal':
                            port.setCausality('Causal')
                            for port2 in element.getPorts():
                                if port2 != port: port2.setCausality('Uncausal')
                        # assigning based on rule of GY, if one port is causal, another must be causal also and vice versa (both uncausal)
                        alter_ports = [an_port for an_port in element.getPorts() if an_port != port and an_port.getCausality() != None]
                        if len(alter_ports) > 0:
                            if alter_ports[0].getCausality() == 'Causal':
                                port.setCausality('Causal')
                            elif alter_ports[0].getCausality() == 'Uncausal':
                                port.setCausality('Uncausal')
                            else:
                                print('Errror in conditions')
                else:
                    pass
                    # print('no info for assigning, skipping for GY')

    def applyTF_GY_rules(self):
        # NOT USED
        # GY-RULE: (only 2 ports must be connected) - power go in from one side and out on another side (power conservative element), 
        # CAUSALITY only for one port, causality can be reversed during auto asignment
        list_of_tf = [element for element in self.getElementsList() if element.getType() == 'GY']
        # IF WE HAVE 
        for item in list_of_tf:
            item_ports = item.getPorts()
            if item_ports[0].getDirection() == 'Input':
                item_ports[1].setDirection('Output')
            elif item_ports[1].getDirection() == 'Output':
                item_ports[0].setDirection('Input')
            else:
#                 print(f'WARNING! UNdefined direction {item_ports}')
                pass

            if item_ports[0].getCausality() == 'Causal':
                item_ports[1].setCausality('Uncausal')
            elif item_ports[1].getCausality() == 'Uncausal':
                item_ports[0].setCausality('Causal')
            else:
#                 print(f'WARNING! UNdefined direction {item_ports}')
                pass
    def update_bondsport_status(self):
        '''
        If one port assigned, the opposite one can be assigned because of common bond
        '''
        ## time to update all ports depended on defined ports (RULE: Bond connects only opposite property value ports)
        port_pairs = []
        for bond in self.getBondList():
            port_pairs.append([bond.getFromPort(), bond.getToPort()])
        
        #update the input-output values
        for pair in port_pairs:
            if pair[0].getDirection() == 'Input':
                pair[1].setDirection('Output')
            if pair[0].getDirection() == 'Output':
                pair[1].setDirection('Input')
            if pair[1].getDirection() == 'Input':
                pair[0].setDirection('Output')
            if pair[1].getDirection() == 'Output':
                pair[0].setDirection('Input')
           
        # update the causal-uncausal values
        for pair in port_pairs:
            if pair[0].getCausality() == 'Causal':
                pair[1].setCausality('Uncausal')
            if pair[0].getCausality() == 'Uncausal':
                pair[1].setCausality('Causal')
            if pair[1].getCausality() == 'Causal':
                pair[0].setCausality('Uncausal')
            if pair[1].getCausality() == 'Uncausal':
                pair[0].setCausality('Causal')

    def apply_one_zero_junction_rule(self):
        # for all 0 and 1 junctions make arrows and causality automaticaly
        list_of_ones = [element for element in self.getElementsList() if element.getType() == '1']
        list_of_zeros = [element for element in self.getElementsList() if element.getType() == '0']

        # POWER ROUTING RULE
        # 1-Junction RULES
        for item in list_of_ones:
            item_ports = item.getPorts()
            state_equation = []
            input_counter = 0
            output_counter = 0
            for port in item_ports:
                if port.getDirection() == 'Input':
                    input_counter += 1
                elif port.getDirection() == 'Output':
                    output_counter += 1
                else:
                    pass
#                     print('None val')

            # RULE! If we have N ports and N-1 ports are Input, the rest one must be Output
            if input_counter == len(item_ports)-1:
                for port in item_ports:
                    if port.getDirection() is None:
                        port.setDirection('Output')
            else:
                if self.debug: print('passing, no info to do...')
                
            # RULE! If we have N ports and N-1 ports are Output, the rest one must be Input
            if output_counter == len(item_ports)-1:
                for port in item_ports:
                    if port.getDirection() is None:
                        port.setDirection('Input')
            else:
                if self.debug: print('passing, no info to do...')

        # 0-Junction RULES
        #TODO 0-junction may have more than 2 inputs and 2 outputs
        for item in list_of_zeros:
            item_ports = item.getPorts()
            state_equation = []
            input_counter = 0
            output_counter = 0
            for port in item_ports:
                if port.getDirection() == 'Input':
                    input_counter += 1
                elif port.getDirection() == 'Output':
                    output_counter += 1
                else:
                    print('NB! None val')
            if input_counter == len(item_ports)-1:
                if self.debug: print('Assign output')
                for port in item_ports:
                    if port.getDirection() is None:
                        port.setDirection('Output')
            else:
                if self.debug: print('passing')

            if output_counter == len(item_ports)-1:
                if self.debug: print('Assign input')
                for port in item_ports:
                    if port.getDirection() is None:
                        port.setDirection('Input')
            else:
                if self.debug: print('passing')

        # Flip bonds "from" and "to" depended on port direction of nodes (useful for visualization)
        for bond in self.getBondList():
            port = bond.getToPort()
            if port.getDirection() == 'Output':
                if self.debug: print('Flip bond directions according input/output ports ...')
                tempo = bond.getToPort()
                bond.setToPort(bond.getFromPort())
                bond.setFromPort(tempo)
            port = bond.getFromPort()
            if port.getDirection() == 'Input':
                if self.debug: print('Flip bond directions according input/output ports ...')
                tempo = bond.getFromPort()
                bond.setFromPort(bond.getToPort())
                bond.setToPort(tempo)

        # CAUSALITY RULE in JUNCTIONS
        # 1-Junction RULES:
        for item in list_of_ones:
            item_ports = item.getPorts()
            state_equation = []
            causal_counter = len([port for port in item_ports if port.getCausality() == 'Causal'])
            uncausal_counter = len([port for port in item_ports if port.getCausality() == 'Uncausal'])
            # if N-1 ports causal, the one last is uncausal
            if causal_counter == len(item_ports)-1:
                if self.debug: print('Assign Uncausal')
                for port in item_ports:
                    if port.getCausality() is None:
                        port.setCausality('Uncausal')
            else:
                if self.debug: print('passing')
            # If one port is Uncausal, then all others Causal
            if uncausal_counter == 1:
                if self.debug: print('Assign causal')
                for port in item_ports:
                    if port.getCausality() is None:
                        port.setCausality('Causal')
            else:
                if self.debug: print('passing')

        # 0-Junction RULE: 1-Junction RULES:
        for item in list_of_zeros:
            item_ports =item.getPorts()
            state_equation = []
            causal_counter = len([port for port in item_ports if port.getCausality() == 'Causal'])
            uncausal_counter = len([port for port in item_ports if port.getCausality() == 'Uncausal'])
            if self.debug and causal_counter == 0: print('Causality is not detectable yet')
            
            # If one port is Causal, then all others Uncausal
            if causal_counter == 1:
                if self.debug: print('Assign uncausal')
                for port in item_ports:
                    if port.getCausality() is None:
                        port.setCausality('Uncausal')
            else:
                if self.debug: print('passing')
            
            # if n-1 PORTS ARE UNCAUSAL, THAN LAST ONE IS CAUSAL
            if uncausal_counter == len(item_ports)-1:
                if self.debug: print('Assign causal')
                for port in item_ports:
                    if port.getCausality() is None:
                        port.setCausality('Causal')
            else:
                if self.debug: print('passing')
    
    def verifyRules(self):
        """
        Originally this method used after "applyRules" methos for model completeness check!
        (or algebraic loops and any other problems).
        self.__elementsList --> model.getElementsList()
        self.__bondsList --> model.getBondList()
        """
        for element in self.__elementsList:
            # print(element.getType(), ports)
            if element.getType() == '1':
                count_causal_ports = 0
                for port in element.getPorts():
                    if port.getCausality() == 'Uncausal':
                        count_causal_ports += 1
                if count_causal_ports > 1:
                    print(f'Element id={element.getId()} ERROR 11----- ONLY ONE Uncausal ALLOWED FOR 1 JUNCTION -----')
            if element.getType() == '0':
                count_uncausal_ports = 0
                for port in element.getPorts():
                    if port.getCausality() == 'Causal':
                        count_uncausal_ports += 1
                if count_uncausal_ports > 1:
                    print(f'Element id={element.getId()} ERROR 12----- ONLY ONE Causal ALLOWED FOR 0 JUNCTION -----')
    
    def applyRules(self):
        """
        
        Iteratively applied the rules to assign BG ports (not bonds) using known data.
        Algebraec loops is not avoided and can lead to errors (be carefull!!!).
        """
        i = 0
        while None in self.display_PortsStatus() or None in self.display_Causality():
            # print('next iteration', i)
            i += 1
            self.assign_sources_ports()
            self.update_bondsport_status()
            self.assign_ports_for_CRI_elemts()
            self.assign_ports_TF_GY()
            self.apply_one_zero_junction_rule()
            if i > 100:
                print('ERROR 001: FAILED! Max iterations reached, while applyRules, probably there are some algebraic loops')
                break
        
    ##### Automatically get the state-space eq #############################################
    @staticmethod
    def express_new(equation, var):
        '''

        equation = '+e4+e7'
        var = '+e7'
        -------
        res = '-e7=+e4'
        '''
        if var[0] == '+':
            var_inv = var.replace('+', '-')
        else:
            var_inv = var.replace('-', '+')
        items = []
        last_item_pos = 0
        for i in range(len(equation)):
            if (equation[i] == '-' or equation[i] == '+') and equation[last_item_pos:i] != '':
                items.append(equation[last_item_pos:i])
                last_item_pos = i
            if i == len(equation) - 1:
                items.append(equation[last_item_pos:i+1]) #last variable
        # if requested var is not found 
        pos = equation.find(var)
        if pos == -1:
            # print('Express: not found the variable, inverse variable check')
            pos_inv = equation.find(var_inv)
            if pos_inv == -1:
                print('ERROR 002: NOT FOUND the variable in initial and inverse equation')
            else:
                items.remove(var_inv)
                res = var + '='
                for item in items:
                    res += item
        else:
            items.remove(var)
            res = var + '='
            # inverse all othe vars
            for item in items:
#                 print(item)
                if item[0] == '+':
                    itemstr = '-' + item[1:]
                elif item[0] == '-':
                    itemstr = '+' + item[1:]
                res += itemstr
        return res

    def assign_0_1_junctions_effort_flow(self):
        '''
        Note! All ports direction and causality must be defined before this step.
        For 0-1 junctions assign all bonds an input or output effort-flows (determine equations 1 step)
        Note: on this step we don't have all required information and some elements e/f will be undefined
        '''
        for element in self.getElementsList():
            if element.getEffort() == None:
            # the element effort is not known still... try to find using connected elements
            # RULE! 1-junction is sum of efforts
                if element.getType() == '1':
                    #the equation will depended on power flow direction
                    # find all connected ports (two connected by bond)
                    connected_ports = []
                    for port in self.getPortsList():
                        if port in element.getPorts():
                            connected_ports.append(port)
                    element_eq = '' #effort
                    cont_eq = '' #flow
                    for port in connected_ports:
                        #i-th port of element
                        for bond in self.getBondList():
                            if port == bond.getFromPort() or port == bond.getToPort():
                                #efforts sum must be equal to 0... signs depended on POWER FLOW DIRECTION
                                if port.getDirection() == 'Input':
                                    element_eq += '+e' + str(bond.getId())
                                elif port.getDirection() == 'Output':
                                    element_eq += '-e' + str(bond.getId())
                                # flows
                                if cont_eq == '':
                                    cont_eq+='+f'+ str(bond.getId())
                                else:
                                    cont_eq+='=+f'+ str(bond.getId())

                    element.setEffort(element_eq)
                    element.setFlow(cont_eq)
                    if self.debug: print('1-junction assigned(effort, flow)=', element_eq, cont_eq)
                if element.getType() == '0':
                    #the equation will depended on causality
                    # find all connected ports
                    connected_ports = []
                    for port in self.getPortsList():
                        if port in element.getPorts():
                            connected_ports.append(port)
                    element_eq = ''
                    cont_eq = ''
                    for port in connected_ports:
                        #i-th port of element
                        for bond in self.getBondList():
                            if port == bond.getFromPort() or port == bond.getToPort():
                                if port.getDirection() == 'Input':
                                    element_eq += '+f' + str(bond.getId())
                                elif port.getDirection() == 'Output':
                                    element_eq += '-f' + str(bond.getId())
                                if cont_eq == '':
                                    cont_eq+='+e'+ str(bond.getId())
                                else:
                                    cont_eq+='=+e'+ str(bond.getId())
                    element.setFlow(element_eq)
                    element.setEffort(cont_eq)
                    if self.debug: print('0-junction assigned(effort, flow)=', element_eq, cont_eq)

    def assign_ports_according_parent(self):
        """
        Just asign the ports according bond signs
        """
        #depricated
        if self.debug: print("assign_ports_according_parent")
        for element in self.getElementsList():
            for port in self.getPortsList():
                if port in element.getPorts():
                    port.effort = element.getEffort()
                    port.flow = element.getFlow()

    def get_model_equations(self):
        '''
        Returns all equations for each effort and flows
        
        equations - list of string elements
        '''
        equations = []
        for bond in self.getBondList():
            if bond.effort == '' or bond.flow == '':
                print("ERROR 007: The bond id:", bond.getId(), " has no effort or flow")
            equations.append(bond.effort)
            equations.append(bond.flow)
        return equations

    def assign_bonds(self):
        '''
        Assign equations for bonds

        for all bonds (except 0 and 1 junction,  that have been defined)
        assign effort-flow equations for all bonds, firstly for bonds connecting to C, I, R, SE, SF, secondly for 1-0 and 0-1 bonds

        '''
        if self.debug: print("assign_bonds...")
        for bond in self.getBondList():
            connected_ports = {}
            for port in self.getPortsList():
                if port == bond.getFromPort():
                    connected_ports[0] = port
                elif port == bond.getToPort():
                    connected_ports[1] = port
        #     print(connected_ports[0], connected_ports[1], '--------------------------------------->>>>')
            if self.debug: print(".....for elements C, I, R, SE, SF") 
            for element in self.getElementsList():
                for element2 in self.getElementsList():
                    if connected_ports[0] in element.getPorts() and connected_ports[1] in element2.getPorts():
                        if self.debug: print('both elements found...id-s:', element.getId(), element2.getId(), ', type: ',element.getType(), element2.getType(), '-----------------------')

                        if element.getType() == '1' and element2.getType() == 'C':
                            #print('from C to 1 detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                bond.effort = '+e' + str(bond.getId()) + '=' + '+1/C' + str(element2.getId()) + '*q' + str(element2.getId())
                                element2.setFlow('+f' + str(bond.getId())) # to be used later for equations system
                                bond.flow = element.getFlow() #self.express(element.getFlow(), '+f' + str(bond.getId()))
                                
                                if self.debug: print('------', element.getFlow())
                                if self.debug: print('-++C', 'effort=', bond.effort, 'flow=', bond.flow)

                        if element.getType() == '1' and element2.getType() == 'I':
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                bond.effort = self.express_new(element.getEffort(), '+e' + str(bond.getId()))
                                bond.flow = '+f' + str(bond.getId()) + '=' + '+1/I' + str(element2.getId()) + '*p' + str(element2.getId())
                                element2.setEffort('+e' + str(bond.getId())) # to be used later for equations system
                                
                                if self.debug: print('-++I', 'effort=',  bond.effort, 'flow=', bond.flow)

                        if element.getType() == '1' and element2.getType() == 'R':
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                bond.effort = '+e' + str(bond.getId()) + '='  + str(element2.getParam()) + '*f' + str(bond.getId())
                                bond.flow = element.getFlow()
                                if self.debug: print('-++R', 'effort=',  bond.effort, 'flow=', bond.flow)
                                
                        if element.getType() == '1' and element2.getType() == 'SF':
                            #print('from 1 to SF detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                if self.debug: print('TRACE::::', element.getEffort(), 'expressing:::::','+e' + str(bond.getId()))
                                bond.effort = self.express_new(element.getEffort(), '+e' + str(bond.getId()))
                                bond.flow = '+f' + str(bond.getId()) + '=' + element2.getFlow()
                                if self.debug: print('-++SF', 'effort=',  bond.effort, 'flow=', bond.flow)
                                
                        if element.getType() == 'SE' and element2.getType() == '1':
                            #print('from SE to 1 detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                bond.effort = '+e' + str(bond.getId())+'=' + element.getEffort()
                                bond.flow = element2.getFlow()
                                if self.debug: print('-++SE', 'effort=',  bond.effort, 'flow=', bond.flow)

                        if element.getType() == '1' and element2.getType() == 'TF':
                            print('from 1 to TF detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
#                             print('causality:', connected_ports[0].getCausality(), connected_ports[1].getCausality())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                print('Fr 1 to TF Bond =', bond.getId(), 'el1 eff=', element.getEffort(), 'el1 flow=', element.getFlow(), 'EL2 eff=', element2.getEffort(), 'el2 flow=', element2.getFlow())
                                #+'=' + element2.getParam() + element2.getEffort()
                                bond.flow = element.getFlow() # obtaining from left side (not TF side)
                                # RULE: the TF's effort is known and flow is not known then partially fill flow
                                if element2.getFlow() == None:
                                    element2.setFlow('+f' + str(bond.getId()) + '*1/' + element2.getParam()) # prepare for other side bond
                                    print('TFs flow prepared: ', element2.getFlow())
                                if element2.getEffort() != None:
                                    # if this is prepeared already when we compleate anouther side of bond
                                    bond.effort = '+e' + str(bond.getId()) + '=' + element2.getEffort()
                                if self.debug: print('-++TF effort=',  bond.effort, 'flow=', bond.flow)
                            else:
                                print('ERROR: detected FROM and To mismatching the Output and Input, troubles with assigning ports')
                        
                        if element.getType() == 'TF' and element2.getType() == '1':
                            print('from TF to 1 detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                print('Fr TF to 1 conn... Bond iD=', bond.getId(), 'elem1 eff=', element.getEffort(), 'el1 flow=', element.getFlow(), 'elem2 eff=', element2.getEffort(), 'elem2 fl=', element2.getFlow())
#                                 bond.effort = self.express_new(element2.getEffort(), '+e' + str(bond.getId()))
                                bond.effort = self.express_new(element2.getEffort(), '+e' + str(bond.getId()))
                                if element.getFlow() != None:
                                    # if this is prepeared already then handled other side bond
                                    bond.flow = '+f' + str(bond.getId()) + '=' + element.getFlow() # use prepeared Flow from other side
                                if element.getEffort() == None:
                                    # RULE: if the TF's effort still is unknown /// then partially fill it (prepare)
                                    element.setEffort('+e' + str(bond.getId()) + '*' + element.getParam()) # prepare effors part for other side bond
                                    print('TF prepare effort', element.getEffort())
                                    
                                if self.debug: print('-+-TF effort=',  bond.effort, 'flow=', bond.flow)
                                if self.debug: print('BOND ID=', bond.getId())
                            else:
                                print('ERROR: detected FROM and To mismatching the Output and Input, troubles with assigning ports')
                        
                        if element.getType() == '1' and element2.getType() == 'GY':
                            if self.debug: print('from 1 to GY detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
#                             print('causality:', connected_ports[0].getCausality(), connected_ports[1].getCausality())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                print('Fr 1 to GY Bond =', bond.getId(), 'el1 eff=', element.getEffort(), 'el1 flow=', element.getFlow(), 'EL2 eff=', element2.getEffort(), 'el2 flow=', element2.getFlow())
                                #+'=' + element2.getParam() + element2.getEffort()
                                bond.flow = element.getFlow() # RULE: 1-junction has common flow (comes not from TF side), hense fill it
                                if element2.getFlow() == None:
                                    element2.setFlow('+f' + str(bond.getId()) + '*' + element2.getParam()) # prepare for other side of GY (=f*m)
                                    print('GY prepared flow', element2.getFlow())
                                if element2.getEffort() != None:
                                    # if this is prepeared already when we compleate anouther side of bond
                                    bond.effort = '+e' + str(bond.getId()) + '=' + element2.getEffort()
                                if self.debug: print('-++GY effort=',  bond.effort, 'flow=', bond.flow)
                            else:
                                print('ERROR: detected FROM and To mismatching the Output and Input, troubles with assigning ports')

                        if element.getType() == 'GY' and element2.getType() == '1':
                            print('from GY to 1 detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                print('Fr GY to 1 conn... Bond iD=', bond.getId(), 'elem1 eff=', element.getEffort(), 'el1 flow=', element.getFlow(), 'elem2 eff=', element2.getEffort(), 'elem2 fl=', element2.getFlow())
#                                 bond.effort = self.express_new(element2.getEffort(), '+e' + str(bond.getId()))
                                bond.flow = element2.getFlow() 
                                if element.getFlow() == None:
                                    # if this is prepeared already then handled other side bond
                                    bond.flow = '+f' + str(bond.getId()) + '=' + element2.getFlow() # use prepeared Flow from other side
                                if element.getEffort() == None:
                                    # RULE: if the GY's effort still is unknown // then partially fill it (prepare)
                                    element.setEffort('+f' + str(bond.getId()) + '*' + element.getParam()) # prepare effors part for other side bond
                                    # print('GY prepare effort', element.getEffort(), element.getEffort())
                                    # bond.effort = '+e' + str(bond.getId()) + '=' + element.getEffort()
                                else:
                                    # already prepared GY effor value
                                    bond.effort = '+e' + str(bond.getId()) + '=' + element.getFlow()

                                print('-+-GY effort=',  bond.effort, 'flow=', bond.flow)
                                print('BOND ID=', bond.getId())
                            else:
                                print('ERROR: detected FROM and To mismatching the Output and Input, troubles with assigning ports')
                        
            # same for 1-0 junctions
            # print(".....for elements 1, 0") 
            for element in self.getElementsList():
                for element2 in self.getElementsList():
                    if connected_ports[0] in element.getPorts() and connected_ports[1] in element2.getPorts():
        #                 print('both elements found...', element.getId(), element2.getId())
                        if element.getType() == '1' and element2.getType() == '0':
        #                     print('from 1 to 0 detected', connected_ports[0].getDirection(), connected_ports[1].getDirection())
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                bond.effort = element2.getEffort() #effort from 0 junction
                                bond.flow = element.getFlow()
                                if self.debug: print('1 to 0 effort=',  bond.effort, 'flow=', bond.flow)
                            #TODO Causality check
                        if element.getType() == '0' and element2.getType() == '1':
                            if connected_ports[0].getDirection() == 'Output' and connected_ports[1].getDirection() == 'Input':
                                if connected_ports[0].getCausality() == 'Causal':
                                    bond.effort = self.express_new(element2.getEffort(), '+e' + str(bond.getId()))
                                    bond.flow = self.express_new(element.getFlow(), '+f' + str(bond.getId()))
                                    if self.debug: print('0 to 1 effors', element.getFlow(), element2.getFlow(),'flows', element.getEffort(), element2.getEffort())
                                    if self.debug: print('bond caus 0-1', 'effort=',  bond.effort, 'flow=', bond.flow)
                                elif connected_ports[1].getCausality() == 'Causal':
                                    bond.effort = self.express_new(element2.getEffort(), '+e'+str(bond.getId()))
                                    bond.flow = element2.getFlow()
                                    if self.debug: print('bond 0-1', 'effort=',  bond.effort, 'flow=', bond.flow)
                                else:
                                    print('ERROR 003, Probably not defined causality...')

    def assign_equations(self):

        '''
        
        NB! Before equations derivation you have to assing all ports and bonds in the model!!!
        Pipeline: Assign system equations (effort-flow for each bond) 
        '''
        not_compleated = True
        i = 0
        while not_compleated and i < 10:    
            self.assign_0_1_junctions_effort_flow()
            self.assign_ports_according_parent()
            self.assign_bonds() 
            i+=1
            if '' not in self.get_model_equations():
                not_compleated = False
        if i == 10:
            print(f'Assigning of equations is FAILED! Max inetration reached: {i}')

        
    @staticmethod
    def express_eq_as_zero(equastions):
        '''
        Express all equations as equal to zero (same form for all)
        
        Return list of string elements
        '''
        equastions_eq_zero = []
        for item in equastions:
            splited = item.split('=')
            #print(splited, len(splited))
            if len(splited) > 2:
                for i in range(0, len(splited)-1):
                    eq = ''
                    if splited[0][0]=='-':
                        eq = splited[0].replace('-', '+')+splited[i+1]
                    elif splited[0][0]=='+':
                        eq = splited[0].replace('+', '-')+splited[i+1]
                    equastions_eq_zero.append(eq)
            else:
                if splited[0][0] == '-':
                    splited[0] = splited[0].replace('-', '+')
                    eq = splited[0] + splited[1]
                elif splited[0][0] == '+':
                    splited[0]= splited[0].replace('+', '-')
                    eq = splited[0] + splited[1]
                else:
                    print('ERROR 004: error during making the equations same form, probably some sign lost')
                equastions_eq_zero.append(eq)
        return equastions_eq_zero
    
    def detect_capasitors_inputs_states(self, equastions_eq_zero):

        '''
        #add the derivatives for energy storing elements
        main components of state equations must be detected before

        Returns:
        capacitors - state equastions (dictionary key is string, value is simpy symbol)
        input_variables - variables which values could be controlled
        state_variables - the name of state variables
        parameter_variables - the name of variables which have initial value
        '''
        capacitor_variables = {} # the key is time derivative of capacitor states
        input_variables = []
        state_variables = []
        parameter_variables = []
        for element in self.getElementsList():
            if element.getType() == 'C':
                capacitor_variables['d' + element.getstateVariable() + '/dt'] = sp.sympify(element.getFlow())
            if element.getType() == 'I':
                capacitor_variables['d' + element.getstateVariable() + '/dt'] = sp.sympify(element.getEffort())
            if element.getstateVariable() is not None:
                state_variables.append(sp.sympify(element.getstateVariable()))
            if element.getinputVariable() is not None:
                input_variables.append(sp.sympify(element.getinputVariable()))
            if element.getParam() is not None:
                parameter_variables.append(sp.sympify(element.getParam()))
        if self.debug: print('capacitor_variables =', capacitor_variables, 'input_variables=', input_variables, 'state_variables=', state_variables, 'parameter_variables=', parameter_variables)
        
        self.capacitor_variables = capacitor_variables
        self.input_variables = input_variables
        self.state_variables = state_variables
        self.parameter_variables = parameter_variables
    
    #@staticmethod
    def simplify_eq_and_get_variables(self, equastions_eq_zero):
        '''
        Calculates the effort flow pararameter list.
        
        Input:
        state variables - p and q variables        
        input_variables - sources e and f
        
        Returns:
        variables - all variables list

        final_vars - variables in final eq
        eff_flows - none state variables list
        ''' 
        
        equastions_sp = []
        variables = []
        final_vars = []
        for eq in equastions_eq_zero:
    #         print(sp.sympify(eq))
            equastions_sp.append(sp.sympify(eq))
            for el in list(sp.sympify(eq).free_symbols):
                variables.append(el)
        
        for j in [self.input_variables, self.state_variables, self.parameter_variables]:
            for i in j:
                final_vars.append(i)

        eff_flows = list(set(variables) - set(final_vars))
        
        self.all_variables = variables
        self.final_variables = final_vars
        self.eff_flows_sp = eff_flows
        self.equastions_sp = equastions_sp
        
#         return variables, final_vars, eff_flows, equastions_sp
    
    def all_defined(self, target_vars, final_vars, defined_variables):
        '''
        Check! was the value found for each variable
        '''
        counter = 0
        for i in target_vars:
            if i in final_vars or i in defined_variables:
                counter+=1
        if counter==len(target_vars):
            return True
        else:
            return False
    
    def not_irreverse(self,target_var, solution_vars, variables_exp):
        '''
        checking is the var in our list of primary vars
        '''
        res = True
        if self.debug: print('not_irreverse cheking..var =', target_var, 'solution vars', solution_vars)
        #target_v e0
        #solutuin_varts e1, f3
        for i in solution_vars:
            if variables_exp[target_var] != None:
                if self.debug: print('not_irreverse cheking none values ..',variables_exp[target_var])
                if target_var in variables_exp[target_var]:
                    res=False
    #     print('////', res, target_var, solution_vars)
        return res
    
    def effort_flow_iteration(self, eff_flows, equastions_sp, final_vars, variables_exp, defined_variables):
        '''
        Returns all variables and values for that variables

        defined_variables -  all expressions found for this vars
        '''
        for target_var in eff_flows:
            result = None
            for expr in equastions_sp:
                if target_var in expr.free_symbols:
                    solution_vars = sp.solve(expr, target_var)[0].free_symbols
                    if self.all_defined(solution_vars, final_vars, defined_variables) and self.not_irreverse(target_var, solution_vars, variables_exp): 
                        result = sp.solve(expr, target_var)
    #                     print(target_var, '---------->', result)
                        break
            if result is None:
                pass
            else:
                if variables_exp[target_var]==None:
                    if self.debug: print('---', target_var, '==================>', result)
                    variables_exp[target_var] = result
                    defined_variables.append(target_var)

        return defined_variables, variables_exp
    
    def dict_has_None(self,variables_exp):
        '''
        Checking has the dictionary a none value 
        '''
        st = False
        for k,v in variables_exp.items():
            if v == None:
                st = True
        return st
    
    def make_final_eq_expressions(self, eff_flows_values, variables_exp):
        '''
        Returns expressions for each effort and flows
        
        
        variables_exp - expression for each variable
        self.eff_flows_sp - 
        eff_flows_values - 
        '''
        defined_variables = []
        while self.dict_has_None(variables_exp):
            defined_variables, variables_exp = self.effort_flow_iteration(self.eff_flows_sp, self.equastions_sp, self.final_variables, variables_exp, defined_variables)
            if self.debug: print('iterates------------------------------>')
        return variables_exp
    
    def non_state_var(self, variables, final_vars):
        '''
        Return True if some variables in final_vars
        or False
        '''
        for var in variables:
            if var not in final_vars:
                return True
        return False
    
    def cauchy_form_state_eq(self, variables_exp):
        '''
        Makes equastions in cauchy form! Not matrix form!
        
        
        Input examples:
            self.capacitor_variables = {'dp4/dt': e4, 'dq2/dt': f2}
            input_variables = [SE0, SF1]
            variables_exp = {e6: [R3*f6], f3: [f2 - f4], f1: [p4/I4], f6: [f3], e3: [-e5 - e6], e2: [e3], f5: [f3], e4: [e2], f4: [f7], f7: [SF1], e5: [q2/C2], e1: [e0 - e2], e7: [-e4], f2: [f0], f0: [f1], e0: [SE0]}
            final_vars = [SE0, SF1, q2, p4, C2, R3, I4]
        return:
            cauchy_form_state_eq - dictionary with equasions
        '''
        cauchy_form_state_eq = {}
        print('capacitor_variables=', self.capacitor_variables)
        print('final_vars=', self.final_variables)
        # build equastions for A-B matrix
        for k, v in self.capacitor_variables.items():
            print(k, '=')
            #find solution for this value
            expression = v
            variables_list = expression.free_symbols
            print('variables list =', variables_list)
            while(self.non_state_var(variables_list, self.final_variables)):
                # while there is no state variables in variable list
                for variable in variables_list:
                    for key, value in variables_exp.items():
                        if variable == key:
                            print('substituting...', key, '-----b---',value[0])
                            expression = expression.subs({key:value[0]})
                            print('exp in progress...',expression)
                variables_list = expression.free_symbols
            cauchy_form_state_eq[k] = expression.expand()
            print(expression)
        return cauchy_form_state_eq
    
    @staticmethod
    def get_variables(input_text):
        '''
        Returns the variables as list from given expression
        
        input example:
        input_text = 'e2*e7/r5 + r2*r7'
        
        output example:
        

        '''
        comps = []
        nobrackets = sp.sympify(input_text)
        if len(nobrackets.args)>1:
            for i in nobrackets.args:
                print(sp.Poly(i, sp.sympify('e2')).all_coeffs())
                comps.append(i.free_symbols)
        return comps

    def make_same_order(self, cauchy_state_equastions, state_variables):
        '''
        makes same order the componens of equation according to the order of state variables.
        '''
        for num, key in enumerate(cauchy_state_equastions):
            while (str(state_variables[num]) != str(key).split('/')[0][1:]):
                temp = state_variables[num]
                state_variables.remove(temp)
                state_variables.append(temp)
        return cauchy_state_equastions, state_variables
    
    @staticmethod
    def has_state(state_var, component):
        '''
        Return True if the component contains the given state variable
        Input:
            state_var - current state var
            state_variables - list
            component - sympy oblect
        '''
        ans = False
        for i in component.free_symbols:
            if state_var == i:
                ans = True
        return ans
    
    def cauchy_form_output_eq(self, variables_exp, output_variables):
        '''
        Returns the equations describing the output of the model, to be used later for C, D matrix
        NB! Requred to enter the variables name of system output.

        output_variables = "e7,f7"
        '''
        print('Input the names of output variables:')
        print('It mast be in list eff_flows: ', self.eff_flows_sp, '(Velocity of Force at some point)')
        #output_variables = input().split(',') # uncomment to take variables from input field
        output_variables = output_variables.split(',')
        print('U entered this: ', output_variables)
        output_variables_simpy = [sp.symbols(v) for v  in output_variables]

        cauchy_form_output_eq = {}
        for var in output_variables_simpy:
            print('output variable= ', var)
            expression = None
            for k, v in enumerate(variables_exp.keys()):
        #         print(k, v)
                if v == var:
                    print("first match", variables_exp[v])
                    expression = variables_exp[v][0]
                    expression = sp.simplify(expression, evaluate=False)
                    variables_list = expression.free_symbols
                    while(self.non_state_var(variables_list, self.final_variables)):
                        # while the re is no state variables in variable list
                        for variable in variables_list:
                            for key, value in variables_exp.items():
                                if variable == key:
                                    expression = expression.subs({key:value[0]})
                        variables_list = expression.free_symbols
                        print('next iteration with expression:', expression)
                    cauchy_form_output_eq[var] = expression.expand()

        return cauchy_form_output_eq
    
    def make_matrix_from_cauchy(self, dummy_matrix, cauchy_state_equastions, variables_vector):
        '''
        Trasforms the state equasions to SimPY matrix form of state-space
        Inputs:
        cauchy_state_equastions - dict of first derivative equastions
        variables_vector - list of either state variables or input_vartiables
        Outputs:
        dummy_matrix - simpy matrix with the result components
        
        '''
        # cauchy_state_equastions, variables_vector = self.make_same_order(cauchy_state_equastions, variables_vector) # make state variables in same order as derivatives... only ONES
        m_size = len(variables_vector)
        
        for eq_id, value in enumerate(cauchy_state_equastions.values()):
            # for each derivatives eq all brackets mast be open before
            value = sp.simplify(value, evaluate=False)
            # print("after simpify value=", value)
            # print("--------------looking for args", value.args)

            # TAKING ARGS WE MUST BE CARIFULL to avoid over separation of components
            if value.func == sp.core.add.Add:
                # print('---------------THIS IS ADD, SO LOOK FOR ARGS---------------')
                for component in value.args:
                    # print('component:', component, 'args=', component.args, '--------------')
                    #iterates over items in equation
                    # print("looking for ", component.free_symbols, 'in ', variables_vector)
                    for state_var_id, state_variable in enumerate(variables_vector):
                        #itrate over states
                        # print("component.free_symbols= ", component.free_symbols)
                        if state_variable in component.free_symbols:
                            # print(f'state var {state_variable} detected in component {component}')
                            nobrackets = component / state_variable
                            # print('to be saved in matrix', nobrackets)
                            dummy_matrix[eq_id, state_var_id] = nobrackets
            else:
                # print("this is not add, e.i. single component of equastion -----------------")
                component = value
                for state_var_id, state_variable in enumerate(variables_vector):
                    #itrate over states
                    # print("component.free_symbols= ", component.free_symbols)
                    if state_variable in component.free_symbols:
                        # print(f'state var {state_variable} detected in component {component}')
                        nobrackets = component / state_variable
                        # print('to be saved in matrix', nobrackets)
                        dummy_matrix[eq_id, state_var_id] = nobrackets

        # print('dummy_matrix=', dummy_matrix)
        return dummy_matrix
    
    def make_state_statespace(self, cauchy_state_equastions):
        '''
        Trasforms the state equasions to state space matrix A and B
        
        Inputs:
        cauchy_state_equastions - dict of first derivative equastions
        self.state_variables - list of state variables name
        self.input_vartiables - the inputs of the system
        
        Outputs:
        state_space_A - matrix sympy
        state_space_B - matrix sympy
        
        '''
        cauchy_state_equastions, self.state_variables = self.make_same_order(cauchy_state_equastions, self.state_variables)
        m_size = len(self.state_variables)
        n_size = len(self.input_variables)

        x_vect = np.array(self.state_variables).reshape((m_size, 1)) # colom vect
        u_vect = np.array(self.input_variables).reshape((n_size, 1)) # colom vect
        # print("x_vect=", x_vect)
        # print("u_vect=", u_vect)
        A = sp.MatrixSymbol('A', m_size, m_size)
        B = sp.MatrixSymbol('B', m_size, n_size)
        A_matrix = sp.Matrix(A)
        B_matrix = sp.Matrix(B)
        # fill mtrx by 0
        for i in range(m_size):
            for j in range(m_size):
                A_matrix[i, j] = 0
        for i in range(m_size):
            for j in range(n_size):
                B_matrix[i, j] = 0
        # for state
        A_matrix = self.make_matrix_from_cauchy(A_matrix, cauchy_state_equastions, self.state_variables)
        # for input
        B_matrix = self.make_matrix_from_cauchy(B_matrix, cauchy_state_equastions, self.input_variables)
        # print(A_matrix, B_matrix)
        self.__A = A_matrix
        self.__B = B_matrix
    
    def make_output_statespace(self, cauchy_form_output_eq):
        '''
        Makes a symbolic C and D matrix in state-space form
        Inputs:
        cauchy_form_output_eq - output equastions
        '''
        m_size = len(self.state_variables)
        n_size = len(self.input_variables)

        C = sp.MatrixSymbol('C', m_size, m_size)
        D = sp.MatrixSymbol('D', m_size, n_size)
        C_matrix = sp.Matrix(C)
        D_matrix = sp.Matrix(D)

        # fill mtrx by 0
        for i in range(m_size):
            for j in range(m_size):
                C_matrix[i, j] = 0
        for i in range(m_size):
            for j in range(n_size):
                D_matrix[i, j] = 0
        # for state
        C_matrix = self.make_matrix_from_cauchy(C_matrix, cauchy_form_output_eq, self.state_variables)
        # for input
        D_matrix = self.make_matrix_from_cauchy(D_matrix, cauchy_form_output_eq, self.input_variables)
        # print(C_matrix, D_matrix)
        
        self.__C = C_matrix
        self.__D = D_matrix
    
    def cauchy_form_equastions_sequence(self):
        ''' 
        Makes a several manipulations to obtain a cauchy form equations
        
        Returns: 
        cauchy_state_equastions - dictionary with state variable derivatives as keys and their values
        variables_exp - dictionary contains the expressions for all variables
        final_vars - all variables in equations
        state_variables - list of state variables
        eff_flows_sp - all effort flow variable names
        input_variables - input_variables
        '''
        self.assign_equations()
        equastions = self.get_model_equations()
        
        equastions_eq_zero = self.express_eq_as_zero(equastions)
        #capacitors, input_variables, state_variables, parameters = 
        self.detect_capasitors_inputs_states(equastions_eq_zero)
#         capacitor_variables, input_variables, state_variables, parameter_variables

        
        self.simplify_eq_and_get_variables(equastions_eq_zero)
        
        # below variables hold  the data with simpy format
        eff_flows_values = [None] * len(self.eff_flows_sp)
        variables_exp = dict(zip(self.eff_flows_sp, eff_flows_values))
        variables_exp = self.make_final_eq_expressions(eff_flows_values, variables_exp)
        cauchy_state_equastions = self.cauchy_form_state_eq(variables_exp)
        return cauchy_state_equastions, variables_exp #, final_vars, state_variables, self.eff_flows_sp, input_variables

    def simulate(self, initial_state, input_sequence, time_steps, sampling_period, parameters_values):
        '''
        Simulates the state-space model using the backward Euler method
        Input:
        -- A,B,C,D              - continuous time system matrices 
        -- initial_state      - the initial state of the system 
        -- time_steps         - the total number of simulation time steps 
        -- sampling_period    - the sampling period for the backward Euler discretization 
        parameters_values     - numerical values
        Returns:
            the state sequence and the output sequence stored in the vectors Xd and Yd respectively

        Input example:
        A = np.array([[ -0.1 , -10.  ],
           [  0.05,   0.  ]])
        B = np.array([[ 0., 20.],
           [ 0.,  0.]])
        C = np.array([[ -0.1 , -10.  ],
           [  0.05,   0.  ]])
        D = np.array([[  0., -20.],
           [  0.,   0.]])
        initial_state = np.array([[0.],
           [0.]])
        time_steps=400
        sampling_period=0.5
        input_sequence=np.ones((time_steps, len(input_variables)))
        '''
        parameters_list = self.parameter_variables 
        # make numerical materix frim symbolic
        A = np.array(self.__A.subs({parameters_list[i]:parameters_values[i] for i in range(len(parameters_list))})).astype(np.float64)
        B = np.array(self.__B.subs({parameters_list[i]:parameters_values[i] for i in range(len(parameters_list))})).astype(np.float64)
        C = np.array(self.__C.subs({parameters_list[i]:parameters_values[i] for i in range(len(parameters_list))})).astype(np.float64)
        D = np.array(self.__D.subs({parameters_list[i]:parameters_values[i] for i in range(len(parameters_list))})).astype(np.float64)
        
        print(type(A), A.shape, type(B), B.shape)

        I = np.identity(A.shape[0]) # this is an identity matrix
        Ad = np.linalg.inv(I - sampling_period * A)
        Bd = Ad.dot(sampling_period * B)
        Xd = np.zeros(shape=(A.shape[0], time_steps + 1))
        Yd = np.zeros(shape=(C.shape[0], time_steps + 1))
        for i in range(0, time_steps):
            if i == 0:
                Xd[:,[i]] = initial_state
    #             print((C@initial_state.reshape(len(state_variables), 1)).shape)
                Yd[:,[i]] = C@initial_state + D@input_sequence[i].reshape(len(self.input_variables),1)
    #             print((Ad@initial_state).shape, (input_sequence[i]).shape)
                x = Ad@initial_state + Bd@input_sequence[i].reshape(len(self.input_variables),1)
    #             print(x.shape)
            else:
                Xd[:,[i]] = x
                Yd[:,[i]] = C@x + D@input_sequence[i].reshape(len(self.input_variables),1)
                x = Ad@x + Bd@input_sequence[i].reshape(len(self.input_variables), 1)
        Xd[:,[-1]] = x
        Yd[:,[-1]]  = C@x + D@input_sequence[i].reshape(len(self.input_variables),1)
        #TODO correctness of D matrix Using should be verified later
        return Xd, Yd

# START DISPLAY FUNCTIONS 
def show_bonds_effort_flow(model:BondGraph):
    '''
    Display effort-flow of bonds
    '''
    for bond in model.getBondList():
        print(f'Bond id {bond.getId()}, effort={bond.effort}, flow={bond.flow}')

def show_ports_state(model:BondGraph):
    '''
    Display elements and ports data: input or output and Causality
    print like a tree
    '''
    for element in model.getElementsList():
        print('Element name: ', element.getName())
        for port in element.getPorts():
            print('++ port_id:', port.getId(), '|  arrow: ', port.getDirection(), '|  causality: ', port.getCausality())

### END DISPLAY FUNCTIONS




#------------------- for next work - additional classes for generalization
# from Extras.enum import Enum

# class BondType(Enum):
#     Signal = (0, 'Signal')
#     Power  = (1, 'Power')

# class CausalityType(Enum):
#     Tail    = (-1, 'Tail')
#     Acausal = ( 0, 'Acausal') # Undefined
#     Head    = ( 1, 'Head')

# class ArrowType(Enum):
#     Tail      = (-1, 'Tail')
#     Undefined = ( 0, 'Undefined')
#     Head      = ( 1, 'Head')
    
# class ElementType(Enum):
#     Junction     = (0,'Junction')
#     Source       = (1,'Source')
#     Storage      = (2,'Storage')
#     Transduction = (2,'Transduction')
#     Dissipation  = (5,'Dissipation')
#     Undefined    = (-1,'Undefined')

# class JunctionType(Enum):
#     Zero = 0 + 2*ElementType.Junction[0]
#     One  = 1 + 2*ElementType.Junction[0]
    
# class SourceType(Enum):
#     Flow   = 0 + 2*ElementType.Source[0]
#     Effort = 1 + 2*ElementType.Source[0]
    
# class StorageType(Enum):
#     Capacitor = 0 + 2*ElementType.Storage[0]
#     Inertia   = 1 + 2*ElementType.Storage[0]

# class TransducerType(Enum):
#     Transformer = 0 + 2*ElementType.Transduction[0]
#     Gyrator     = 1 + 2*ElementType.Transduction[0]
    
# class DissipationType(Enum):
#     Resistance = 0 + 2*ElementType.Dissipation[0]
#     Admittance = 1 + 2*ElementType.Dissipation[0]
    