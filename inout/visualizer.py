# io/visualizer.py
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import math


import matplotlib.pyplot as plt
import networkx as nx




def find_element_by_port(model, port):
    """Возвращает BGElement, которому принадлежит данный порт."""
    for element in model.elements:
        if port in element.ports:
            return element
    return None

def custom_layered_layout(model, distance=2.0, angle_step=math.pi/10, check_tol=0.15, max_iters=100):
    """
    previus vertion @render function

    Располагает новые узлы на фиксированном расстоянии от связанных,
    выбирая угол с максимальным расстоянием до других размещённых узлов.
    После расчёта координаты назначаются в .position всех элементов модели.
    """
    adj = model.adjacency_dict()
    coords = {}
    placed = set()

    # Сохраняем уже размещённые позиции
    for elem in model.elements:
        if elem.position and None not in elem.position:
            coords[elem.id] = tuple(elem.position)
            placed.add(elem.id)

    # Стартовый (центр) — если ничего не размещено, берём самый связанный
    if not placed:
        center_node = max(adj, key=lambda k: len(adj[k]))
        coords[center_node] = (0.0, 0.0)
        placed.add(center_node)
        queue = [center_node]
    else:
        queue = list(placed)

    assigned = set(placed)
    while queue:
        node = queue.pop(0)
        x0, y0 = coords[node]
        neighbors = adj[node]
        unassigned = [n for n in neighbors if n not in assigned]
        if not unassigned:
            continue

        n_angles = max(len(unassigned), 4)
        angles_full_circle = [2 * math.pi * i / n_angles for i in range(n_angles)]

        for nbr in unassigned:
            # Ищем угол с максимальным расстоянием до всех остальных, кроме node
            best_angle = None
            best_min_dist = -1
            best_xy = None

            for angle in angles_full_circle:
                r = distance
                for try_r in [distance, distance * 1.2, distance * 1.4]:
                    x = x0 + try_r * math.cos(angle)
                    y = y0 + try_r * math.sin(angle)
                    # Считаем расстояния до уже размещённых (кроме node, к которому крепим)
                    min_dist = min(
                        math.hypot(x - xx, y - yy)
                        for eid, (xx, yy) in coords.items() if eid != node
                    ) if len(coords) > 1 else float('inf')

                    # Проверяем, достаточно ли далеко (больше check_tol)
                    if min_dist > best_min_dist and min_dist > check_tol:
                        best_min_dist = min_dist
                        best_angle = angle
                        best_xy = (x, y)
                # после первого успешного — не крутим r дальше

            if best_xy is None:
                # Не нашли удачного угла — увеличиваем радиус, ищем "пустое место"
                angle = angles_full_circle[0]
                for i in range(1, max_iters):
                    r = distance + i * 0.5
                    x = x0 + r * math.cos(angle)
                    y = y0 + r * math.sin(angle)
                    min_dist = min(
                        math.hypot(x - xx, y - yy)
                        for eid, (xx, yy) in coords.items() if eid != node
                    ) if len(coords) > 1 else float('inf')
                    if min_dist > check_tol:
                        best_xy = (x, y)
                        break
                else:
                    # Если вообще нет места — ставим рядом (но это очень редко!)
                    best_xy = (x0 + distance, y0)
            coords[nbr] = best_xy
            assigned.add(nbr)
            queue.append(nbr)

    # === НАЗНАЧАЕМ КООРДИНАТЫ В МОДЕЛЬ ===
    for elem in model.elements:
        pos = coords.get(elem.id, None)
        if pos:
            elem.position = [pos[0], pos[1]]



def draw_bond_graph(model):
    """
    Отрисовывает граф с уже заданными позициями.
    """
    nodes = [element.id for element in model.elements]
    real_edges = []
    unreal_edges = []
    port_pairs = []

    for bond in model.bonds:
        port_pairs.append([bond.from_port, bond.to_port])

    # Реальные связи (Power flow)
    for port_pair in port_pairs:
        bond_from = bond_to = None
        for element in model.elements:
            if port_pair[0] in element.ports:
                bond_from = element.id
            if port_pair[1] in element.ports:
                bond_to = element.id
        real_edges.append((bond_from, bond_to))

        # Causal (Unreal) edges
        bond_from = bond_to = None
        for element in model.elements:
            if port_pair[0] in element.ports:
                if port_pair[0].causality == 'Uncausal':
                    bond_to = element.id
                elif port_pair[0].causality == 'Causal':
                    bond_from = element.id
            if port_pair[1] in element.ports:
                if port_pair[1].causality == 'Causal':
                    bond_from = element.id
                elif port_pair[1].causality == 'Uncausal':
                    bond_to = element.id
        if bond_from is not None and bond_to is not None:
            unreal_edges.append((bond_from, bond_to))

    edge_labels = {ed: i for i, ed in enumerate(real_edges)}

    # Координаты — берем из element.position
    pos = {element.id: tuple(element.position) for element in model.elements}

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(real_edges)

    plt.figure(figsize=(19, 13))
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color="tab:red", node_size=1200)
    nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color='b', arrowsize=18, arrows=True, connectionstyle='arc3, rad=0.2', min_source_margin=23, min_target_margin=23)
    nx.draw_networkx_edges(G, pos, edgelist=unreal_edges, edge_color='g', arrows=True, connectionstyle='angle3, angleA=90, angleB=0', arrowstyle=']-', min_source_margin=20, min_target_margin=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Кастомные подписи к узлам
    for elem in model.elements:
        x, y = pos[elem.id]
        plt.text(x - 0.2, y + 0.25, str(elem.icon), color='blue', fontsize=9, fontweight='normal', ha='center', va='center')
        plt.text(x, y, str(elem.name), color='white', fontsize=12, fontweight='bold', ha='center', va='center')
        if getattr(elem, "value", None) not in (None, ''):
            plt.text(x + 0.2, y - 0.35, str(elem.value), color='green', fontsize=10, fontweight='bold', ha='center', va='center')

    plt.margins(0.2)
    plt.title(f"Bond Graph: {getattr(model, 'name', '')}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

