# bondgraph.py (или rules/engine.py)
from bond_graph_simulation.core.BondGraph2 import BondGraphModel
from bond_graph_simulation.core.base import ElementType, BGPort, BGBond

class RuleEngine:
    def __init__(self, model: BondGraphModel, debug=False):
        self.model = model
        self.debug = debug

    def apply_all(self):
        """
        Ineractively applies rules:
        
        1. assign_sources_ports()
        2. update_bondsport_status() (flip bonds if needed)
        3. assign_ports_for_CRI_elements()
        4. assign_ports_TF_GY()
        5. apply_one_zero_junction_rule()

        Untill all ports have direction/causality assigned or max_iter reached.
        """
        max_iter = 100
        for iteration in range(max_iter):
            changed = False

            changed |= self.assign_sources_ports()
            changed |= self.update_bondsport_status()
            changed |= self.assign_ports_for_CRI_elements()
            changed |= self.update_bondsport_status()
            changed |= self.assign_ports_TF_GY()
            changed |= self.apply_one_junction_rule()
            changed |= self.apply_zero_junction_rule()

            if self.is_complete():
                if self.debug:
                    print(f"[INFO] converge after {iteration+1} iteration(s)")
                break

            if not changed and not self.is_complete():
                # no progress but still incomplete => might be conflict
                if self.debug:
                    print("[WARNING] No more progress, but incomplete ports.")
                break
        else:
            print("[ERROR] Reached max iteration in apply_all, some ports may remain None")

        # Optional final check
        self.verify_causality_and_direction()

    def find_invalid_bonds(self):
        """
        Checks for invalid bonds in the model:
        1. Is there a bond, where both ports have the same direction.
        2. For all bonds, except those connecting TF/GY, is there a bond, where both ports have the same causality.
        3. #TODO: Add more checks if needed.
        Returns a list of problematic bonds with a description of the problem.
        """
        problems = []
        for bond in self.model.bonds:
            port1 = bond.from_port
            port2 = bond.to_port

            # Find elements that used these ports
            elem1 = next((e for e in self.model.elements if any(p.id == port1.id for p in e.ports)), None)
            elem2 = next((e for e in self.model.elements if any(p.id == port2.id for p in e.ports)), None)

            # 1. Check direction
            if port1.direction is not None and port2.direction is not None and port1.direction == port2.direction:
                problems.append({
                    "bond_id": bond.id,
                    "problem": "Both ports have the same direction",
                    "direction": port1.direction,
                    "elements": (elem1.name if elem1 else '?', elem2.name if elem2 else '?')
                })

            # 2. Check causality (except TF and GY)
            tf_gy_types = {'TF', 'GY'}
            # Check types of both elements; if at least one is not TF/GY — apply rule
            elem1_type = elem1.type.value if elem1 else None
            elem2_type = elem2.type.value if elem2 else None
            if not (elem1_type in tf_gy_types or elem2_type in tf_gy_types):
                if port1.causality is not None and port2.causality is not None and port1.causality == port2.causality:
                    problems.append({
                        "bond_id": bond.id,
                        "problem": "Both ports have the same causality (for non-TF/GY)",
                        "causality": port1.causality,
                        "elements": (elem1.name if elem1 else '?', elem2.name if elem2 else '?')
                    })

        return problems
    
    def assign_sources_ports(self):
        """
        Automatically assigns direction and causality for sources (SE/SF).
        
        SE: direction=Output, causality=Uncausal (output)
        SF: direction=Input,  causality=Causal    (input)
        
        + Flip bond without resetting direction/causality of ports.
        + direction and causality assined independently.
        """
        changed = False
        for element in self.model.elements:
            # SE (source of effort)
            if element.type == ElementType.SOURCE_EFFORT:
                if self.debug:
                    print('SE ports =', element.ports)
                for port in element.ports:
                    if self.debug: 
                        print('port==', port, 'directions----<>>>', port.direction, port.causality)
                    old_dir, old_caus = port.direction, port.causality

                    # direction
                    if port.direction is None:
                        for bond in self.model.bonds:
                            if bond.from_port.id == port.id:
                                port.direction = 'Output'
                                changed = True
                                if self.debug:
                                    print(old_dir, old_caus , 'chnged to -->', port.direction, port.causality)
                                break
                            elif bond.to_port.id == port.id:
                                if self.debug:
                                    print('Source of effort cant have power input, Redefining..')
                                # Flip bond
                                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                                port.direction = 'Output'
                                changed = True
                                if self.debug:
                                    print(old_dir, old_caus , 'chnged to -->', port.direction, port.causality)
                                break

                    # causality
                    if port.causality is None:
                        for bond in self.model.bonds:
                            if bond.from_port.id == port.id or bond.to_port.id == port.id:
                                port.causality = 'Uncausal'
                                changed = True
                                if self.debug:
                                    print(f"port {port.id}: causality set to Uncausal")
                                break

            # SF (source of flow)
            elif element.type == ElementType.SOURCE_FLOW:
                if self.debug:
                    print('SF ports =', element.ports)
                for port in element.ports:
                    if self.debug:
                        print('port==', port, 'directions----<>>>', port.direction, port.causality)
                    old_dir, old_caus = port.direction, port.causality

                    # direction
                    if port.direction is None:
                        for bond in self.model.bonds:
                            if bond.to_port.id == port.id:
                                port.direction = 'Input'
                                changed = True
                                if self.debug:
                                    print(old_dir, old_caus , 'chnged to -->', port.direction, port.causality)
                                break
                            elif bond.from_port.id == port.id:
                                if self.debug:
                                    print('Source of flow cant have power output, Redefining..')
                                # Flip bond
                                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                                port.direction = 'Input'
                                changed = True
                                print(old_dir, old_caus , 'chnged to -->', port.direction, port.causality)
                                break

                    # causality
                    if port.causality is None:
                        for bond in self.model.bonds:
                            if bond.from_port.id == port.id or bond.to_port.id == port.id:
                                port.causality = 'Causal'
                                changed = True
                                if self.debug:
                                    print(f"port {port.id}: causality set to Causal")
                                break
        return changed

    # 2) Sinchronize bonds, flip if conflicted
    def update_bondsport_status(self):
        """
        If one port of the bond has direction or causality assigned,
        the second port gets the opposite value automatically.
        Bond always connects opposite property value ports.
        """
        changed = False
        for bond in self.model.bonds:
            port1 = bond.from_port
            port2 = bond.to_port

            # --- Input/Output direction ---
            if port1.direction == 'Input' and port2.direction is None:
                port2.direction = 'Output'
                changed = True
            if port1.direction == 'Output' and port2.direction is None:
                port2.direction = 'Input'
                changed = True
            if port2.direction == 'Input' and port1.direction is None:
                port1.direction = 'Output'
                changed = True
            if port2.direction == 'Output' and port1.direction is None:
                port1.direction = 'Input'
                changed = True

            # --- Causal/Uncausal ---
            if port1.causality == 'Causal' and port2.causality is None:
                port2.causality = 'Uncausal'
                changed = True
            if port1.causality == 'Uncausal' and port2.causality is None:
                port2.causality = 'Causal'
                changed = True
            if port2.causality == 'Causal' and port1.causality is None:
                port1.causality = 'Uncausal'
                changed = True
            if port2.causality == 'Uncausal' and port1.causality is None:
                port1.causality = 'Causal'
                changed = True

        return changed
    # 3) R/C/I

    def assign_ports_for_CRI_elements(self):
        """
        Assigns direction and causality for C, R, I elements.
        C and R: direction=Input, causality=Uncausal
        I:     direction=Input, causality=Causal
        Flip bond, if port is not on the right side.
        """
        changed = False
        for element in self.model.elements:
            # Capacitor (C)
            if element.type == ElementType.CAPACITOR:
                if self.debug:
                    print('C ports =', element.ports)
                for port in element.ports:
                    if port.direction is None and port.causality is None:
                        for bond in self.model.bonds:
                            if bond.from_port.id == port.id:
                                if self.debug:
                                    print('Capacitor cant have power input, Redefining..')
                                # Flip bond
                                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                                port.direction = 'Input'
                                port.causality = 'Uncausal'
                                changed = True
                            elif bond.to_port.id == port.id:
                                port.direction = 'Input'
                                port.causality = 'Uncausal'
                                changed = True
            # Resistor (R)
            elif element.type == ElementType.RESISTOR:
                if self.debug:
                    print('R ports =', element.ports)
                for port in element.ports:
                    if port.direction is None and port.causality is None:
                        for bond in self.model.bonds:
                            if bond.from_port.id == port.id:
                                if self.debug:
                                    print('Dissipative element cant have power flow out, Redefining..')
                                # Flip bond
                                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                                port.direction = 'Input'
                                port.causality = 'Uncausal'
                                changed = True
                            elif bond.to_port == port:
                                port.direction = 'Input'
                                port.causality = 'Uncausal'
                                changed = True
            # Inductor (I)
            elif element.type == ElementType.INDUCTOR:
                if self.debug:
                    print('I ports =', element.ports)
                for port in element.ports:
                    if port.direction is None and port.causality is None:
                        for bond in self.model.bonds:
                            if bond.from_port == port:
                                if self.debug:
                                    print('I-element cant have power go out in linear form')
                                # Flip bond
                                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                                port.direction = 'Input'
                                port.causality = 'Causal'
                                changed = True
                            elif bond.to_port == port:
                                port.direction = 'Input'
                                port.causality = 'Causal'
                                changed = True
        return changed
    # -------------------------------------------------------------------------
    # 4) TF / GY
    # -------------------------------------------------------------------------
    
    def assign_ports_TF_GY(self):
        """

        Assigns direction and causality for TF and GY (two-port power conservative elements).
        TF: one Input, one Output; one Causal, one Uncausal
        GY: one Input, one Output; both Causal or both Uncausal
        RULE: Assign only if one port has been determined due to connected elements. 
        """
        changed = False

        # TRANSFORMER (TF)
        for element in [e for e in self.model.elements if e.type == ElementType.TRANSFORMER]:
            if len(element.ports) != 2:
                continue

            p1, p2 = element.ports

            # ---- direction ----
            # if both ports Unknown — skip
            if p1.direction is None and p2.direction is None:
                pass
            # if both known — the second must be opposite (change now the second)
            elif p1.direction is not None and p2.direction is None:
                p2.direction = 'Input' if p1.direction == 'Output' else 'Output'
                changed = True
            elif p2.direction is not None and p1.direction is None:
                p1.direction = 'Input' if p2.direction == 'Output' else 'Output'
                changed = True

            # ---- causality ----
            # TF must have opposite causality
            if p1.causality is None and p2.causality is None:
                # both None --> skip
                pass
            elif p1.causality is not None and p2.causality is None:
                p2.causality = 'Uncausal' if p1.causality == 'Causal' else 'Causal'
                changed = True
            elif p2.causality is not None and p1.causality is None:
                p1.causality = 'Uncausal' if p2.causality == 'Causal' else 'Causal'
                changed = True

        # GYRATOR (GY)
        # -------- GYRATOR (GY) --------
        for element in [e for e in self.model.elements if e.type == ElementType.GYRATOR]:
            if len(element.ports) != 2:
                continue

            p1, p2 = element.ports

            # --- Direction ---
            # same as for TF: must be opposite
            if p1.direction is None and p2.direction is None:
                pass
            elif p1.direction is not None and p2.direction is None:
                p2.direction = 'Input' if p1.direction == 'Output' else 'Output'
                changed = True
            elif p2.direction is not None and p1.direction is None:
                p1.direction = 'Input' if p2.direction == 'Output' else 'Output'
                changed = True

            # --- Causality ---
            # GY: both must be same
            if p1.causality is None and p2.causality is None:
                pass
            elif p1.causality is not None and p2.causality is None:
                p2.causality = p1.causality
                changed = True
            elif p2.causality is not None and p1.causality is None:
                p1.causality = p2.causality
                changed = True

        return changed

    # 5) Junction 0/1
    def apply_one_junction_rule(self):
        """
        Automatically assigns direction and causality for 1-junctions according to BondGraph rules.
        Automatically assigns direction and causality for 1-junctions according to BondGraph rules.
        Flip bonds. 
        TODO CLean all prints
        """
        changed = False
        # list of 1-junctions
        ones = [e for e in self.model.elements if e.type == ElementType.JUNCTION_ONE]

        # === 1-Junction (flow direction rules)
        for item in ones:
            item_ports = item.ports
            input_counter = sum(1 for port in item_ports if port.direction == 'Input')
            output_counter = sum(1 for port in item_ports if port.direction == 'Output')
            if self.debug: print(f"1-junction {item.name}: input_ports={input_counter}, output_ports={output_counter}")

            # If N-1 ports has Input, last one make Output
            if input_counter == len(item_ports) - 1:
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Output'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: порт {port.id} назначен Output")
                    else:
                        if self.debug: print(f"1-junction {item.name}: direction уже назначен для порта {port.id} ({port.direction}). Предупреждение: direction problem!!!!")
            # else:
            #     if self.debug: print(f'1-junction {item.name}: passing, no info to do...')

            # If N-1 ports Output, last one make Input
            elif output_counter == len(item_ports) - 1:
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Input'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: порт {port.id} назначен Input")
                    else:
                        if self.debug: print(f"1-junction {item.name}: direction уже назначен для порта {port.id} ({port.direction})Предупреждение: direction problem 2!!!!")
            # else:
            #     if self.debug: print(f'1-junction {item.name}: passing, no info to do...')

            # If 1 input already exists, then make others output
            elif input_counter >= 1:
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Output'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: port ID= {port.id} changed Output")
                    else:
                        if self.debug: print(f"1-junction {item.name}: direction already assigned for port {port.id} ({port.direction}). Warning: direction problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing, no info to do...')

        # Flip bonds according direction/casuality rules (for visualisation needed at least)
        # kinda sinchronize bonds with ports
        for bond in self.model.bonds:
            port_to = bond.to_port
            # If port_to became Output — flip bond direction
            if port_to.direction == 'Output':
                if self.debug: print(f'Flip bond {bond.id} (to_port {port_to.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: NOW from_port={bond.from_port.id}, to_port={bond.to_port.id}")
            port_from = bond.from_port
            # If port_from became Input — flip bond direction
            if port_from.direction == 'Input':
                if self.debug: print(f'Flip bond {bond.id} (from_port {port_from.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: NOW from_port={bond.from_port.id}, to_port={bond.to_port.id}")

        # Assign causality for 1-junction
        for item in ones:
            item_ports = item.ports
            causal_counter = sum(1 for port in item_ports if port.causality == 'Causal')
            uncausal_counter = sum(1 for port in item_ports if port.causality == 'Uncausal')
            if self.debug: print(f"1-junction {item.name}: causal_ports={causal_counter}, uncausal_ports={uncausal_counter}")

            # If N-1 ports Causal, last one Uncausal
            if causal_counter == len(item_ports) - 1:
                if self.debug: print(f'1-junction {item.name}: Assign Uncausal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Uncausal'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: port ID= {port.id} changed Uncausal")
                    else:
                        if self.debug: print(f"1-junction {item.name}: causality already assigned for port {port.id} ({port.causality}). Warning: causality problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing (causality)')

            # If one Uncausal, others Causal
            if uncausal_counter == 1:
                if self.debug: print(f'1-junction {item.name}: Assign causal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Causal'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: port ID= {port.id} changed Causal")
                    else:
                        if self.debug: print(f"1-junction {item.name}: causality already assigned for port {port.id} ({port.causality}). Warning: causality problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing (causality)')

        return changed

    def apply_zero_junction_rule(self):
        """
        Automatically assigns direction and causality for 0-junctions according to BondGraph rules.
        Flip bonds for visualization. All print statements are for tracing.
        """
        changed = False
        # List of 0-junctions
        zeros = [e for e in self.model.elements if e.type == ElementType.JUNCTION_ZERO]

        # 0-Junction (flow direction rules) 
        for item in zeros:
            item_ports = item.ports
            input_counter = sum(1 for port in item_ports if port.direction == 'Input')
            output_counter = sum(1 for port in item_ports if port.direction == 'Output')
            if self.debug: print(f"0-junction {item.name}: input_ports={input_counter}, output_ports={output_counter}")

            # If N-1 ports Input, last one make Output
            if input_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign output')
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Output'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: port ID= {port.id} changed Output")
                    else:
                        if self.debug: print(f"0-junction {item.name}: direction already assigned for port {port.id} ({port.direction})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing')

            # If N-1 ports Output, last one make Input
            if output_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign input')
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Input'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: port ID= {port.id} changed Input")
                    else:
                        if self.debug: print(f"0-junction {item.name}: direction already assigned for port {port.id} ({port.direction})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing')

        # Flip bonds according direction/casuality rules (for visualisation needed at least)
        # Kinda sinchronize bonds with ports
        for bond in self.model.bonds:
            port_to = bond.to_port
            # Если порт назначения стал Output — значит, нужно flip'нуть направление bond'а
            if port_to.direction == 'Output':
                if self.debug: print(f'Flip bond {bond.id} (to_port {port_to.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: NOW from_port={bond.from_port.id}, to_port={bond.to_port.id}")
            port_from = bond.from_port
            # Если порт источника стал Input — тоже нужно flip'нуть bond
            if port_from.direction == 'Input':
                if self.debug: print(f'Flip bond {bond.id} (from_port {port_from.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: NOW from_port={bond.from_port.id}, to_port={bond.to_port.id}")

        # Assign causality for 0-junction
        for item in zeros:
            item_ports = item.ports
            causal_counter = sum(1 for port in item_ports if port.causality == 'Causal')
            uncausal_counter = sum(1 for port in item_ports if port.causality == 'Uncausal')
            if self.debug: print(f"0-junction {item.name}: causal_ports={causal_counter}, uncausal_ports={uncausal_counter}")

            if self.debug and causal_counter == 0: print(f'0-junction {item.name}: Causality is not detectable yet')

            # If one Causal, others Uncausal
            if causal_counter == 1:
                if self.debug: print(f'0-junction {item.name}: Assign uncausal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Uncausal'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: port ID= {port.id} changed Uncausal")
                    else:
                        if self.debug: print(f"0-junction {item.name}: causality already assigned for port {port.id} ({port.causality})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing (causality)')

            # If N-1 ports Uncausal, last one Causal
            if uncausal_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign causal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Causal'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: port ID= {port.id} changed Causal")
                    else:
                        if self.debug: print(f"0-junction {item.name}: causality already assigned for port {port.id} ({port.causality})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing (causality)')

        return changed

    # Helpers

    def is_complete(self):
        """
        Return True, all ports in model have assigned direction and causality.
        Otherwise — False.
        """
        for element in self.model.elements:
            for port in element.ports:
                if port.direction is None or port.causality is None:
                    return False
        return True


    def verify_causality_and_direction(self):
        """
        Checks if all ports in the model have assigned direction and causality.
        If there are unassigned ports — prints a warning and their list.
        If all is ok — prints info (if self.debug).
        """
        missing = []
        for element in self.model.elements:
            for port in element.ports:
                if port.direction is None or port.causality is None:
                    missing.append((element.name, port.name))
        if missing:
            print("[WARNING] Some ports remain unassigned:", missing)
        else:
            if getattr(self, "debug", False):
                print("[INFO] All ports have direction & causality assigned.")

