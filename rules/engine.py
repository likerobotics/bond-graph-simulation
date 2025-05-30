# bondgraph.py (или rules/engine.py)
from core.BondGraph2 import BondGraphModel
from core.base import ElementType, BGPort, BGBond

class RuleEngine:
    def __init__(self, model: BondGraphModel, debug=False):
        self.model = model
        self.debug = debug

    def apply_all(self):
        """
        Итеративно применяет правила:
        1. assign_sources_ports()
        2. update_bondsport_status() (flip bonds if needed)
        3. assign_ports_for_CRI_elements()
        4. assign_ports_TF_GY()
        5. apply_one_zero_junction_rule()
        До тех пор, пока не останется None в direction/causality или достигнут max_iter.
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
        Проверяет:
        1. Есть ли bond, у которого оба порта имеют одинаковый direction.
        2. Для всех bond'ов, кроме соединяющих TF/GY, есть ли bond, у которого оба порта имеют одинаковый causality.
        Возвращает список проблемных bond'ов с описанием проблемы.
        """
        problems = []
        for bond in self.model.bonds:
            port1 = bond.from_port
            port2 = bond.to_port

            # Найти элементы, к которым относятся порты
            elem1 = next((e for e in self.model.elements if any(p.id == port1.id for p in e.ports)), None)
            elem2 = next((e for e in self.model.elements if any(p.id == port2.id for p in e.ports)), None)

            # 1. Проверка direction
            if port1.direction is not None and port2.direction is not None and port1.direction == port2.direction:
                problems.append({
                    "bond_id": bond.id,
                    "problem": "Both ports have the same direction",
                    "direction": port1.direction,
                    "elements": (elem1.name if elem1 else '?', elem2.name if elem2 else '?')
                })

            # 2. Проверка causality (кроме TF и GY)
            tf_gy_types = {'TF', 'GY'}
            # Проверяем типы обоих элементов; если хотя бы один не TF/GY — применяем правило
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

    # -------------------------------------------------------------------------
    # 1) Источники
    # -------------------------------------------------------------------------

    
    def assign_sources_ports(self):
        """
        Автоматически назначает direction и causality для источников (SE/SF).
        SE: direction=Output, causality=Uncausal (выход)
        SF: direction=Input,  causality=Causal    (вход)
        + Flip bond без сброса direction/causality у портов.
        + direction и causality назначаются независимо!
        TODO Очистить от принтов
        """
        changed = False
        for element in self.model.elements:
            # === SE (источник усилия) ===
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

            # === SF (источник потока) ===
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




    # -------------------------------------------------------------------------
    # 2) Синхронизация bond’ов, flip при конфликте
    # -------------------------------------------------------------------------
    def update_bondsport_status(self):
        """
        Если у bond один порт уже назначен (direction или causality),
        второй порт можно назначить автоматически.
        Bond всегда соединяет противоположные property value порты.
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


    # -------------------------------------------------------------------------
    # 3) R/C/I
    # -------------------------------------------------------------------------
    def assign_ports_for_CRI_elements(self):
        """
        Назначает direction и causality для C, R, I элементов.
        C и R: direction=Input, causality=Uncausal
        I:     direction=Input, causality=Causal
        Flip bond, если порт не на нужной стороне.
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
        Назначает direction и causality для TF и GY (двухпортовые power conservative элементы).
        TF: один Input, один Output; один Causal, один Uncausal
        GY: один Input, один Output; оба Causal или оба Uncausal
        """
        changed = False

        # TRANSFORMER (TF)
        tf_elements = [element for element in self.model.elements if element.type == ElementType.TRANSFORMER]
        for element in tf_elements:
            if self.debug: print('TF ports =', element.ports)
            if len(element.ports) != 2:
                continue  # Только для двухпортовых
            port1, port2 = element.ports

            # ---- Назначаем direction ----
            for port in (port1, port2):
                if port.direction is None:
                    # Ищем связь (bond), в которой участвует этот порт
                    connected_bonds = [bond for bond in self.model.bonds if bond.from_port.id == port.id or bond.to_port.id == port.id]
                    if not connected_bonds:
                        continue
                    bond = connected_bonds[0]
                    # Разворачиваем bond если port оказался не на той стороне
                    if bond.to_port.id == port.id:
                        if self.debug: print('TF-element: direction flip for port', port)
                        bond.from_port, bond.to_port = bond.to_port, bond.from_port
                        port.direction = 'Input'
                        # другой порт Output
                        for other_port in element.ports:
                            if other_port != port:
                                other_port.direction = 'Output'
                        changed = True
                    elif bond.from_port.id == port.id:
                        port.direction = 'Output'
                        for other_port in element.ports:
                            if other_port != port:
                                other_port.direction = 'Input'
                        changed = True

            # ---- Назначаем causality ----
            for port in (port1, port2):
                if port.causality is None:
                    connected_bonds = [bond for bond in self.model.bonds if bond.from_port.id == port.id or bond.to_port.id == port.id]
                    if not connected_bonds:
                        continue
                    bond = connected_bonds[0]
                    # Если у второго порта уже есть causality, делаем противоположную
                    for other_port in element.ports:
                        if other_port != port and other_port.causality is not None:
                            port.causality = 'Uncausal' if other_port.causality == 'Causal' else 'Causal'
                            changed = True
                            break
                    else:
                        # Иначе назначаем первому causal, второму uncausal
                        port.causality = 'Causal'
                        for other_port in element.ports:
                            if other_port != port and other_port.causality is None:
                                other_port.causality = 'Uncausal'
                        changed = True

        # GYRATOR (GY)
        gy_elements = [element for element in self.model.elements if element.type == ElementType.GYRATOR]
        for element in gy_elements:
            if self.debug: print('GY ports =', element.ports)
            if len(element.ports) != 2:
                continue
            port1, port2 = element.ports

            # ---- Назначаем direction ----
            for port in (port1, port2):
                if port.direction is None:
                    connected_bonds = [bond for bond in self.model.bonds if bond.from_port.id == port.id or bond.to_port.id == port.id]
                    if not connected_bonds:
                        continue
                    bond = connected_bonds[0]
                    if bond.to_port.id == port.id:
                        bond.from_port, bond.to_port = bond.to_port, bond.from_port
                        port.direction = 'Input'
                        for other_port in element.ports:
                            if other_port != port:
                                other_port.direction = 'Output'
                        changed = True
                    elif bond.from_port.id == port.id:
                        port.direction = 'Output'
                        for other_port in element.ports:
                            if other_port != port:
                                other_port.direction = 'Input'
                        changed = True

            # ---- Назначаем causality ----
            for port in (port1, port2):
                if port.causality is None:
                    connected_bonds = [bond for bond in self.model.bonds if bond.from_port.id == port.id or bond.to_port.id == port.id]
                    if not connected_bonds:
                        continue
                    bond = connected_bonds[0]
                    # Если у второго порта уже есть causality, делаем такое же
                    for other_port in element.ports:
                        if other_port != port and other_port.causality is not None:
                            port.causality = other_port.causality
                            changed = True
                            break
                    else:
                        # Если ни у кого не назначено, назначаем causal обоим
                        port.causality = 'Causal'
                        for other_port in element.ports:
                            if other_port != port and other_port.causality is None:
                                other_port.causality = 'Causal'
                        changed = True

        return changed

    # -------------------------------------------------------------------------
    # 5) Junction 0/1
    # -------------------------------------------------------------------------
    def apply_one_junction_rule(self):
        """
        Автоматически назначает direction и causality для 1-junctions по правилам BondGraph.
        Flip bond'ов. Везде присутствуют print для трассировки.
        """
        changed = False
        # Список всех 1-junction
        ones = [e for e in self.model.elements if e.type == ElementType.JUNCTION_ONE]

        # === 1-Junction (правила направления потока) ===
        for item in ones:
            item_ports = item.ports
            input_counter = sum(1 for port in item_ports if port.direction == 'Input')
            output_counter = sum(1 for port in item_ports if port.direction == 'Output')
            if self.debug: print(f"1-junction {item.name}: input_ports={input_counter}, output_ports={output_counter}")

            # Если N-1 порт Input, последний делаем Output
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

            # Если N-1 порт Output, последний делаем Input
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

            # Если 1 input уже есть, то остальные делаем output
            elif input_counter >= 1:
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Output'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: порт {port.id} назначен Output")
                    else:
                        if self.debug: print(f"1-junction {item.name}: direction уже назначен для порта {port.id} ({port.direction}). Предупреждение: direction problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing, no info to do...')

        # === Flip bonds для корректной визуализации ===
        for bond in self.model.bonds:
            port_to = bond.to_port
            # Если порт назначения стал Output — значит, нужно flip'нуть направление bond'а
            if port_to.direction == 'Output':
                if self.debug: print(f'Flip bond {bond.id} (to_port {port_to.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: теперь from_port={bond.from_port.id}, to_port={bond.to_port.id}")
            port_from = bond.from_port
            # Если порт источника стал Input — тоже нужно flip'нуть bond
            if port_from.direction == 'Input':
                if self.debug: print(f'Flip bond {bond.id} (from_port {port_from.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: теперь from_port={bond.from_port.id}, to_port={bond.to_port.id}")

        # === Назначение причинности для 1-junction ===
        for item in ones:
            item_ports = item.ports
            causal_counter = sum(1 for port in item_ports if port.causality == 'Causal')
            uncausal_counter = sum(1 for port in item_ports if port.causality == 'Uncausal')
            if self.debug: print(f"1-junction {item.name}: causal_ports={causal_counter}, uncausal_ports={uncausal_counter}")

            # Если N-1 порт Causal, последний Uncausal
            if causal_counter == len(item_ports) - 1:
                if self.debug: print(f'1-junction {item.name}: Assign Uncausal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Uncausal'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: порт {port.id} назначен Uncausal")
                    else:
                        if self.debug: print(f"1-junction {item.name}: causality уже назначен для порта {port.id} ({port.causality}) Предупреждение: causality problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing (causality)')

            # Если один Uncausal, остальные Causal
            if uncausal_counter == 1:
                if self.debug: print(f'1-junction {item.name}: Assign causal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Causal'
                        changed = True
                        if self.debug: print(f"1-junction {item.name}: порт {port.id} назначен Causal")
                    else:
                        if self.debug: print(f"1-junction {item.name}: causality уже назначен для порта {port.id} ({port.causality}) Предупреждение: causality problem!!!!")
            else:
                if self.debug: print(f'1-junction {item.name}: passing (causality)')

        return changed

    def apply_zero_junction_rule(self):
        """
        Автоматически назначает direction и causality для 0-junctions по правилам BondGraph.
        Flip bond'ов для визуализации. Везде присутствуют print для трассировки.
        """
        changed = False
        # Список всех 0-junction
        zeros = [e for e in self.model.elements if e.type == ElementType.JUNCTION_ZERO]

        # === 0-Junction (правила направления потока) ===
        for item in zeros:
            item_ports = item.ports
            input_counter = sum(1 for port in item_ports if port.direction == 'Input')
            output_counter = sum(1 for port in item_ports if port.direction == 'Output')
            if self.debug: print(f"0-junction {item.name}: input_ports={input_counter}, output_ports={output_counter}")

            # Если N-1 порт Input, последний делаем Output
            if input_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign output')
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Output'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: порт {port.id} назначен Output")
                    else:
                        if self.debug: print(f"0-junction {item.name}: direction уже назначен для порта {port.id} ({port.direction})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing')

            # Если N-1 порт Output, последний делаем Input
            if output_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign input')
                for port in item_ports:
                    if port.direction is None:
                        port.direction = 'Input'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: порт {port.id} назначен Input")
                    else:
                        if self.debug: print(f"0-junction {item.name}: direction уже назначен для порта {port.id} ({port.direction})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing')

        # === Flip bonds для корректной визуализации ===
        for bond in self.model.bonds:
            port_to = bond.to_port
            # Если порт назначения стал Output — значит, нужно flip'нуть направление bond'а
            if port_to.direction == 'Output':
                if self.debug: print(f'Flip bond {bond.id} (to_port {port_to.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: теперь from_port={bond.from_port.id}, to_port={bond.to_port.id}")
            port_from = bond.from_port
            # Если порт источника стал Input — тоже нужно flip'нуть bond
            if port_from.direction == 'Input':
                if self.debug: print(f'Flip bond {bond.id} (from_port {port_from.id}) directions according input/output ports ...')
                bond.from_port, bond.to_port = bond.to_port, bond.from_port
                changed = True
                if self.debug: print(f"Bond {bond.id} flipped: теперь from_port={bond.from_port.id}, to_port={bond.to_port.id}")

        # === Назначение причинности для 0-junction ===
        for item in zeros:
            item_ports = item.ports
            causal_counter = sum(1 for port in item_ports if port.causality == 'Causal')
            uncausal_counter = sum(1 for port in item_ports if port.causality == 'Uncausal')
            if self.debug: print(f"0-junction {item.name}: causal_ports={causal_counter}, uncausal_ports={uncausal_counter}")

            if self.debug and causal_counter == 0: print(f'0-junction {item.name}: Causality is not detectable yet')

            # Если один Causal, остальные Uncausal
            if causal_counter == 1:
                if self.debug: print(f'0-junction {item.name}: Assign uncausal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Uncausal'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: порт {port.id} назначен Uncausal")
                    else:
                        if self.debug: print(f"0-junction {item.name}: causality уже назначен для порта {port.id} ({port.causality})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing (causality)')

            # Если N-1 порт Uncausal, последний Causal
            if uncausal_counter == len(item_ports) - 1:
                if self.debug: print(f'0-junction {item.name}: Assign causal')
                for port in item_ports:
                    if port.causality is None:
                        port.causality = 'Causal'
                        changed = True
                        if self.debug: print(f"0-junction {item.name}: порт {port.id} назначен Causal")
                    else:
                        if self.debug: print(f"0-junction {item.name}: causality уже назначен для порта {port.id} ({port.causality})")
            else:
                if self.debug: print(f'0-junction {item.name}: passing (causality)')

        return changed


    # -------------------------------------------------------------------------
    # Вспомогательные
    # -------------------------------------------------------------------------
    def is_complete(self):
        """
        Возвращает True, если у всех портов модели назначены и direction, и causality.
        Иначе — False.
        """
        for element in self.model.elements:
            for port in element.ports:
                if port.direction is None or port.causality is None:
                    return False
        return True


    def verify_causality_and_direction(self):
        """
        Проверяет, у всех ли портов в модели назначены direction и causality.
        Если есть незаполненные порты — выводит предупреждение и их список.
        Если всё ок — пишет info (если self.debug).
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

