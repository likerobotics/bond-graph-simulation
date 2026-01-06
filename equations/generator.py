# equations/generator.py
import sympy as sp
from bond_graph_simulation.core.BondGraph2 import BondGraphModel
from bond_graph_simulation.core.base import ElementType, BGPort, BGBond

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

        # Step 1: bond-level eq (R, C, I, SE, SF)
        self._assign_bond_equations()

        #FOR  TF, GY
        self._assign_tf_gy_equations()

        # Step 2: junction eq (0, 1).
        self._assign_junction_equations()

        if self.debug:
            print("[EquationGenerator] Generated equations:")
            for eq in self.equations:
                print("  ", eq)
        return self.equations
    
    ### HELPERS
    def _find_bond_by_port(self, port):
        """
        
        Finds the first bond where one of the ends is the given port.
        Asuming each port is connected to exactly one bond (IT IS RULE!!!).

        """
        for bond in self.model.bonds:
            if bond.from_port is port or bond.to_port is port:
                return bond
        return None
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

    def _assign_tf_gy_equations(self):
        """
        For each Transformer (TF) and Gyrator (GY) element, create the corresponding equations between two bonds.

        For example :
          - TF has parameter n:
                e1 = n * e2
                f2 = n * f1
          - GY has parameter m:
                e1 = m * f2
                e2 = m * f1

        Where:
          b1 ↔ ports[0],  b2 ↔ ports[1].
        """

        for elem in self.model.elements:
            # --- TRANSFORMER ---
            if elem.type == ElementType.TRANSFORMER:
                ports = elem.ports
                if len(ports) != 2:
                    # if self.debug:
                    print(f"[WARNING] TF {elem.name} has {len(ports)} ports, Expected ONLY 2. Skipping!")
                    continue

                b1 = self._find_bond_by_port(ports[0])
                b2 = self._find_bond_by_port(ports[1])
                if b1 is None or b2 is None:
                    if self.debug:
                        print(f"[WARNING] TF {elem.name}: not found bonds for ports. Skipping!")
                    continue

                e1 = sp.Symbol(f"e{b1.id}")
                f1 = sp.Symbol(f"f{b1.id}")
                e2 = sp.Symbol(f"e{b2.id}")
                f2 = sp.Symbol(f"f{b2.id}")

                n = sp.Symbol(elem.parameter)  # n_i
                # Effort relation: e1 = n * e2
                self.equations.append(sp.Eq(e1, n * e2))
                # Flow relation:   f2 = n * f1
                self.equations.append(sp.Eq(f2, n * f1))
                if self.debug:
                    print(f"[TF] {elem.name}: bonds {b1.id} <-> {b2.id}: e1=n*e2, f2=n*f1")

            # --- GYRATOR ---
            elif elem.type == ElementType.GYRATOR:
                ports = elem.ports
                if len(ports) != 2:
                    if self.debug:
                        print(f"[WARNING] GY {elem.name} has {len(ports)} ports, Expected ONLY 2. Skipping.")
                    continue

                b1 = self._find_bond_by_port(ports[0])
                b2 = self._find_bond_by_port(ports[1])
                if b1 is None or b2 is None:
                    if self.debug:
                        print(f"[WARNING] GY {elem.name}: not found bonds for ports. Skipping.")
                    continue

                e1 = sp.Symbol(f"e{b1.id}")
                f1 = sp.Symbol(f"f{b1.id}")
                e2 = sp.Symbol(f"e{b2.id}")
                f2 = sp.Symbol(f"f{b2.id}")

                m = sp.Symbol(elem.parameter)  # m_i

                # Gyrator relations:
                # e1 = m * f2
                # e2 = m * f1
                self.equations.append(sp.Eq(e1, m * f2))
                self.equations.append(sp.Eq(e2, m * f1))

                if self.debug:
                    print(f"[GY] {elem.name}: bonds {b1.id} <-> {b2.id}: e1=m*f2, e2=m*f1")

    # -------------------------------------------------------
    # Part 2: junction eq
    # -------------------------------------------------------

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

        # --- MASTER EFFORT ---
        master_bond = None
        for b in bond_list:
            port = self._which_port_of_element(junction_elem, b)
            if port.causality == 'Output':
                master_bond = b
                break
        if master_bond is None:
            master_bond = bond_list[0]

        master_e = sp.Symbol(f"e{master_bond.id}")

        # --- 1) Equal efforts ---
        for b in bond_list:
            if b is master_bond:
                continue
            e_i = sp.Symbol(f"e{b.id}")
            self.equations.append(sp.Eq(e_i, master_e))

        # --- 2) Sum(flows) = 0  (NO direction signs here!) ---
        flow_sum = 0
        for b in bond_list:
            port = self._which_port_of_element(junction_elem, b)
            sigma = +1 if port.direction == 'Input' else -1
            f_b = sp.Symbol(f"f{b.id}")
            flow_sum += sigma * f_b

        self.equations.append(sp.Eq(flow_sum, 0))


    def _assign_one_junction_eq(self, junction_elem):
        bond_list = self._find_bonds_for_element(junction_elem)
        if len(bond_list) < 2:
            return

        # --- Find master using causality ---
        master_bond = None
        for b in bond_list:
            port = self._which_port_of_element(junction_elem, b)

            # If junction --> Output --> junction giving flow
            if port.causality == 'Output':
                master_bond = b
                break

        if master_bond is None:
            master_bond = bond_list[0]

        master_f = sp.Symbol(f"f{master_bond.id}")

        # --- 1) f_i = f_master
        for b in bond_list:
            if b is master_bond:
                continue
            f_i = sp.Symbol(f"f{b.id}")
            self.equations.append(sp.Eq(f_i, master_f))

        # --- 2) Balance of efforts ---
        effort_sum = 0
        for b in bond_list:
            port = self._which_port_of_element(junction_elem, b)
            e_b = sp.Symbol(f"e{b.id}")
            sigma = +1 if port.direction == 'Input' else -1
            effort_sum += sigma * e_b

        self.equations.append(sp.Eq(effort_sum, 0))

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
