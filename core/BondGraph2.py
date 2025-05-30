# core/graph.py
from typing import List
from core.base import BGElement, BGPort, BGBond, ElementType
from itertools import count
from typing import Optional
from enum import Enum

class BondGraphModel:
    def __init__(self, name: str = "BondGraph"):
        self.name = name
        self.elements: List[BGElement] = []
        self.ports: List[BGPort] = []
        self.bonds: List[BGBond] = []

    def connect(self, from_element: BGElement, to_element: BGElement):
        # Add elements if not already in the model
        if from_element not in self.elements:
            self.elements.append(from_element)
        if to_element not in self.elements:
            self.elements.append(to_element)

        # Always create new ports for each connection
        from_port = BGPort()
        to_port = BGPort()

        from_element.add_port(from_port)
        to_element.add_port(to_port)
        self.ports.extend([from_port, to_port])

        # Create and store bond
        bond = BGBond(from_port, to_port)
        self.bonds.append(bond)
        return bond

    def get_element_by_name(self, name: str):
        for element in self.elements:
            if element.name == name:
                return element
        return None
    
    def find_element_by_port_name(self, port_name):
        """
        Возвращает элемент (BGElement), который содержит порт с заданным именем port_name.
        Если не найден — возвращает None.
        """
        for element in self.elements:
            for port in element.ports:
                if port.name == port_name:
                    return element
        return None
    
    def summary(self):
        print(f"Bond Graph: {self.name}")
        print(f"  Elements: {len(self.elements)}")
        print(f"  Ports: {len(self.ports)}")
        print(f"  Bonds: {len(self.bonds)}")
        
    def adjacency_dict(model):
        """
        Returns the adjacency list representation of the BondGraphModel.
        Does not care about directions. Used only for drawing/position calculation.
        """
        # Собираем id всех элементов
        nodes = [element.id for element in model.elements]
        adj = {node: [] for node in nodes}

        # Для каждого bond ищем, какие элементы соединяются через from_port/to_port
        for bond in model.bonds:
            element_from = None
            element_to = None
            # Ищем элементы, к которым относятся порты
            for element in model.elements:
                if bond.from_port in element.ports:
                    element_from = element.id
                if bond.to_port in element.ports:
                    element_to = element.id
            if element_from is not None and element_to is not None:
                # Добавляем двустороннюю связь
                adj[element_from].append(element_to)
                adj[element_to].append(element_from)
        return adj
    
    def debug_display_bonds(self):
        for bond in self.bonds:
            # Найти элементы, связанные с bond.from_port и bond.to_port
            from_elem = next((e for e in self.elements if bond.from_port in e.ports), None)
            to_elem = next((e for e in self.elements if bond.to_port in e.ports), None)
            print(f"Bond {bond.id}:")
            print(f"  FROM: {from_elem.name if from_elem else '???'} [{bond.from_port.name}]"
                f" (direction={bond.from_port.direction}, causality={bond.from_port.causality})")
            print(f"  TO:   {to_elem.name if to_elem else '???'} [{bond.to_port.name}]"
                f" (direction={bond.to_port.direction}, causality={bond.to_port.causality})")
            print("-" * 40)
            
    def debug_display_elements_with_ports(self):
        for elem in self.elements:
            for port in elem.ports:
                print(f"{elem.name}: Port {port.name} -> direction={port.direction}, causality={port.causality}")