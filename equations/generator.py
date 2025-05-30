# equations/generator.py
import sympy as sp
from core.BondGraph2 import BondGraphModel
from core.base import ElementType, BGPort, BGBond

class EquationGenerator:
    """
    Builds a list of sympy.Eq for each bond and for each junction,
    respecting direction (Input/Output) and causality (Causal/Uncausal).
    These equations typically have the form:
      - eX - R*fX=0  (for resistor)
      - eX - q/C=0   (for capacitor)
      - fX - p/I=0   (for inductor)
      - eX - SE0=0   (for source of effort)
      - fX - SF1=0   (for source of flow)
      - 0-junction: eX are equal, sum_of_flows=0 with sign
      - 1-junction: fX are equal, sum_of_efforts=0 with sign
    The final list is stored in self.equations.
    """

    def __init__(self, model: BondGraphModel, debug=False):
        self.model = model
        self.debug = debug
        self.equations = []

    def generate_equations(self):
        """
        Main entry point:
         1) Clear previous eq
         2) For each bond => build eqs for R, C, I, SE, SF
         3) For each 0/1-junction => build eq for sum of flows/efforts
        Return the list of sympy.Eq
        """
        self.equations.clear()

        # Step 1: bond-level eq (R, C, I, SE, SF, possibly TF, GY).
        self._assign_bond_equations()

        # Step 2: junction eq (0, 1).
        self._assign_junction_equations()

        if self.debug:
            print("[EquationGenerator] Generated equations:")
            for eq in self.equations:
                print("  ", eq)
        return self.equations

    # -------------------------------------------------------
    # Part 1: bond-level eq
    # -------------------------------------------------------
    def _assign_bond_equations(self):
        """
        For each bond, check which element is at from_port / to_port.
        Then build eq like eX - R*fX=0, or eX - q/C=0, etc.
        If bond connects R, we do eX=R*fX, etc.
        If it connects SE, we do eX=SE0, etc.
        Because direction can be reversed, we see which side is 'Input' or 'Output'.
        """
        for bond in self.model.bonds:
            # We'll create symbolic eX, fX for this bond
            e_sym = sp.Symbol(f"e{bond.id}")
            f_sym = sp.Symbol(f"f{bond.id}")

            # Identify which elements are connected
            elem_from = self._find_element_by_port(bond.from_port)
            elem_to = self._find_element_by_port(bond.to_port)
            # It's possible that bond connects two distinct elements or the same in weird cases

            # We'll build eqs for whichever side is R, C, I, SE, SF, etc.
            # Potentially we do multiple eq if both sides are R/C/I, but typically one side is a junction or something else.

            # from_elem
            self._handle_element_side(elem_from, e_sym, f_sym, bond)
            # to_elem
            self._handle_element_side(elem_to, e_sym, f_sym, bond)

    def _handle_element_side(self, elem, e_sym, f_sym, bond):
        """
        Given that 'elem' is connected to this bond,
        if it's R => e_sym - R*f_sym=0,
        if it's C => e_sym - q/C=0,
        if it's I => f_sym - p/I=0, etc.
        """
        if elem is None:
            return

        t = elem.type
        if t == ElementType.RESISTOR:
            # eX - R*fX = 0
            R = sp.Symbol(elem.parameter)  # e.g. R3
            eq = sp.Eq(e_sym, R * f_sym)
            self.equations.append(eq)

        elif t == ElementType.CAPACITOR:
            # eX - q/C = 0
            C = sp.Symbol(elem.parameter)       # e.g. C2
            q = sp.Symbol(elem.state_variable)  # e.g. q2
            eq = sp.Eq(e_sym, q / C)
            self.equations.append(eq)

        elif t == ElementType.INDUCTOR:
            # fX - p/I = 0
            I = sp.Symbol(elem.parameter)
            p = sp.Symbol(elem.state_variable)
            eq = sp.Eq(f_sym, p / I)
            self.equations.append(eq)

        elif t == ElementType.SOURCE_EFFORT:
            # eX - SE0=0
            # if e.effort=+SE0 => name= 'SE0'
            se_name = elem.effort.lstrip('+') if elem.effort else "SE_???"
            se_sym = sp.Symbol(se_name)
            eq = sp.Eq(e_sym, se_sym)
            self.equations.append(eq)

        elif t == ElementType.SOURCE_FLOW:
            # fX - SF1=0
            sf_name = elem.flow.lstrip('+') if elem.flow else "SF_???"
            sf_sym = sp.Symbol(sf_name)
            eq = sp.Eq(f_sym, sf_sym)
            self.equations.append(eq)

        elif t == ElementType.TRANSFORMER:
            # Typically eX = n * eOther or fX= (1/n)*fOther => but that's more advanced
            # For now we skip or partially implement
            pass

        elif t == ElementType.GYRATOR:
            # same logic
            pass

    # -------------------------------------------------------
    # Part 2: junction eq
    # -------------------------------------------------------
    def _assign_junction_equations(self):
        """
        0-junction => all e(bonds) are equal, sum_of_flows=0 with sign
        1-junction => all f(bonds) are equal, sum_of_efforts=0 with sign
        The sign is determined by port.direction=Input => +, Output => -
        """
        for element in self.model.elements:
            if element.type == ElementType.JUNCTION_ZERO:
                self._assign_zero_junction_eq(element)
            elif element.type == ElementType.JUNCTION_ONE:
                self._assign_one_junction_eq(element)

    def _assign_junction_equations(self):
        """
        Generates equations for all 0- and 1-junctions:
        0-junction: all efforts (e) are equal, the sum of flows is zero (considering signs based on direction)  
        1-junction: all flows (f) are equal, the sum of efforts is zero (considering signs based on direction)
        """
        for element in self.model.elements:
            if element.type == ElementType.JUNCTION_ZERO:
                self._assign_zero_junction_eq(element)
            elif element.type == ElementType.JUNCTION_ONE:
                self._assign_one_junction_eq(element)

    def _assign_zero_junction_eq(self, junction_elem):
        bond_list = self._find_bonds_for_element(junction_elem)
        if len(bond_list) < 2:
            return

        # 1) Все e равны: e1 = e2 = ... = en
        e_syms = [sp.Symbol(f"e{b.id}") for b in bond_list]
        # Сделаем одно уравнение: e1 - e2 = 0, e2 - e3 = 0, ... (или можно все к первому)
        for i in range(1, len(e_syms)):
            eq_e = sp.Eq(e_syms[i], e_syms[0])
            self.equations.append(eq_e)
            # Альтернатива (можно добавить одной строкой):
            # eq_e = sp.Eq(sp.Equality(*e_syms), True)

        # 2) Сумма потоков с учетом знака равна нулю
        flow_sum = 0
        for i, b in enumerate(bond_list):
            port_j = self._which_port_of_element(junction_elem, b)
            f_b = sp.Symbol(f"f{b.id}")
            if port_j.direction == 'Input':
                flow_sum += f_b
            else:
                flow_sum -= f_b
        eq_flow = sp.Eq(flow_sum, 0)
        self.equations.append(eq_flow)
        if self.debug:
            print(f"[0-junction] Efforts equalities: {[sp.Eq(e, e_syms[0]) for e in e_syms[1:]]}")
            print(f"[0-junction] Flow sum equation: {eq_flow}")

    def _assign_one_junction_eq(self, junction_elem):
        bond_list = self._find_bonds_for_element(junction_elem)
        if len(bond_list) < 2:
            return

        # 1) Все f равны: f1 = f2 = ... = fn
        f_syms = [sp.Symbol(f"f{b.id}") for b in bond_list]
        for i in range(1, len(f_syms)):
            eq_f = sp.Eq(f_syms[i], f_syms[0])
            self.equations.append(eq_f)
            print(eq_f)

        # 2) Сумма усилий с учетом знака равна нулю
        effort_sum = 0
        for i, b in enumerate(bond_list):
            port_j = self._which_port_of_element(junction_elem, b)
            e_b = sp.Symbol(f"e{b.id}")
            if port_j.direction == 'Input':
                effort_sum += e_b
            else:
                effort_sum -= e_b
        eq_effort = sp.Eq(effort_sum, 0)
        self.equations.append(eq_effort)
        if self.debug:
            print(f"[1-junction] Flows equalities: {[sp.Eq(f, f_syms[0]) for f in f_syms[1:]]}")
            print(f"[1-junction] Effort sum equation: {eq_effort}")

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------
    def _find_element_by_port(self, port):
        for e in self.model.elements:
            if port in e.ports:
                return e
        return None

    def _which_port_of_element(self, elem, bond):
        """
        Return the port belonging to 'elem' that is used in 'bond'
        """
        if bond.from_port in elem.ports:
            return bond.from_port
        elif bond.to_port in elem.ports:
            return bond.to_port
        return None

    def _find_bonds_for_element(self, elem):
        """
        Return list of bonds that attach to any port of elem
        """
        bonds = []
        for b in self.model.bonds:
            if b.from_port in elem.ports or b.to_port in elem.ports:
                bonds.append(b)
        return bonds
