# core/base.py
from itertools import count
from typing import Optional
from enum import Enum


class ElementType(Enum):
    RESISTOR = 'R'
    CAPACITOR = 'C'
    INDUCTOR = 'I'
    SOURCE_EFFORT = 'SE'
    SOURCE_FLOW = 'SF'
    JUNCTION_ZERO = '0'
    JUNCTION_ONE = '1'
    TRANSFORMER = 'TF'
    GYRATOR = 'GY'


class BGElement:
    _id_gen = count()

    def __init__(self, type_: ElementType, name: Optional[str] = None, position=None):
        self.id = next(self._id_gen)
        self.type = type_
        self.name = name or f"{type_.value}_{self.id}"
        self.position = position or [None, None]
        self.value = 0
        self.icon = type_.value
        self.ports = []

        # System-specific attributes
        self.parameter = None
        self.state_variable = None
        self.input_variable = None
        self.effort = None
        self.flow = None

        if type_ == ElementType.INDUCTOR:
            self.parameter = f"I{self.id}"
            self.state_variable = f"p{self.id}"
        elif type_ == ElementType.CAPACITOR:
            self.parameter = f"C{self.id}"
            self.state_variable = f"q{self.id}"
        elif type_ == ElementType.RESISTOR:
            self.parameter = f"R{self.id}"
        elif type_ == ElementType.SOURCE_EFFORT:
            self.input_variable = f"SE{self.id}"
            self.effort = f"+SE{self.id}"
        elif type_ == ElementType.SOURCE_FLOW:
            self.input_variable = f"SF{self.id}"
            self.flow = f"+SF{self.id}"
        elif type_ == ElementType.TRANSFORMER:
            self.parameter = f"n{self.id}"
        elif type_ == ElementType.GYRATOR:
            self.parameter = f"m{self.id}"

    def add_port(self, port):
        self.ports.append(port)


class BGPort:
    _id_gen = count()

    def __init__(self, name=None, direction=None, causality=None):
        self.id = next(self._id_gen)
        self.name = name or f"Port_{self.id}"
        self.direction = direction  # 'Input' or 'Output'
        self.causality = causality  # 'Causal' or 'Uncausal'
        self.effort = None
        self.flow = None


class BGBond:
    _id_gen = count()

    def __init__(self, from_port: BGPort, to_port: BGPort, bond_type='Power'):
        self.id = next(self._id_gen)
        self.from_port = from_port
        self.to_port = to_port
        self.bond_type = bond_type  # e.g., 'Power', 'Signal'
        self.effort = ''
        self.flow = ''


class ElementFactory:
    @staticmethod
    def create(type_str: str, name: Optional[str] = None, position=None) -> BGElement:
        try:
            element_type = ElementType(type_str)
        except ValueError:
            raise ValueError(f"Unknown element type: {type_str}")

        return BGElement(type_=element_type, name=name, position=position)