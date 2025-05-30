import networkx as nx
import matplotlib.pyplot as plt
import random

# Словарь для хранения счётчиков узлов по типам
node_counters = {
    "G*": 0,
    "Y*": 0,
    "SE*": 0,
    "1*": 0,
    "I*": 0,
    "SF*": 0,
    "0*": 0,
    "D*": 0,
    "X*": 0,
    "Z*": 0,
    "R*": 0,
    "C*": 0,
    #"M*": 0,
}

def generate_unique_name_for_type(node_type):
    """Генерирует уникальное имя для узла в графе на основе его типа."""
    global node_counters
    node_counters[node_type] += 1
    return f"{node_type[:-1]}_{node_counters[node_type]}"  # Убираем '*' из типа

# Создаём подграф для узла Y
def create_subgraph_for_Y():
    subgraph = nx.DiGraph()
    node_1_name = generate_unique_name_for_type("1*")
    node_I_name = generate_unique_name_for_type("I*")
    subgraph.add_node(node_1_name, type="1*")
    subgraph.add_node(node_I_name, type="I*")
    subgraph.add_edge(node_1_name, node_I_name)
    return subgraph, "1*"

# Создаём подграф для узла G
def create_subgraph_for_G():
    subgraph = nx.DiGraph()
    node_SF_name = generate_unique_name_for_type("SF*")
    node_1_name = generate_unique_name_for_type("1*")
    node_X_name = generate_unique_name_for_type("X*")
    subgraph.add_node(node_SF_name, type="SF*")
    subgraph.add_node(node_1_name, type="1*")
    subgraph.add_node(node_X_name, type="X*")
    subgraph.add_edge(node_SF_name, node_1_name)
    subgraph.add_edge(node_1_name, node_X_name)
    return subgraph, "X*"

# Создаём первый вариант подграфа для X: 0 -> D
def create_subgraph_for_X_variant_1():
    subgraph = nx.DiGraph()
    node_0_name = generate_unique_name_for_type("0*")
    node_D_name = generate_unique_name_for_type("D*")
    subgraph.add_node(node_0_name, type="0*")
    subgraph.add_node(node_D_name, type="D*")
    subgraph.add_edge(node_0_name, node_D_name)
    return subgraph, "0*"

# Создаём второй вариант подграфа для X: X -> Z
def create_subgraph_for_X_variant_2():
    subgraph = nx.DiGraph()
    node_X_name = generate_unique_name_for_type("X*")
    node_Z_name = generate_unique_name_for_type("Z*")
    subgraph.add_node(node_X_name, type="X*")
    subgraph.add_node(node_Z_name, type="Z*")
    subgraph.add_edge(node_X_name, node_Z_name)
    return subgraph, None

# Универсальная функция для создания подграфа X с выбором варианта
def create_subgraph_for_X():
    variant = random.choice([1, 2])
    if variant == 1:
        return create_subgraph_for_X_variant_1()
    else:
        return create_subgraph_for_X_variant_2()

def create_subgraph_for_Z():
    subgraph = nx.DiGraph()
    node_Y_name = generate_unique_name_for_type("Y*")
    node_X_name = generate_unique_name_for_type("X*")
    subgraph.add_node(node_Y_name, type="Y*")
    subgraph.add_node(node_X_name, type="X*")
    subgraph.add_edge(node_Y_name, node_X_name)
    return subgraph, None

# Создаём подграфы для узла D

def create_subgraph_for_D_variant_1():
    subgraph = nx.DiGraph()
    node_1_name = generate_unique_name_for_type("1*")
    node_C_name = generate_unique_name_for_type("C*")
    subgraph.add_node(node_1_name, type="1*")
    subgraph.add_node(node_C_name, type="C*")
    subgraph.add_edge(node_1_name, node_C_name)
    return subgraph, "1*"

def create_subgraph_for_D_variant_2():
    subgraph = nx.DiGraph()
    node_1_name = generate_unique_name_for_type("1*")
    node_R_name = generate_unique_name_for_type("R*")
    subgraph.add_node(node_1_name, type="1*")
    subgraph.add_node(node_R_name, type="R*")
    subgraph.add_edge(node_1_name, node_R_name)
    return subgraph, "1*"

def create_subgraph_for_D_variant_3():
    subgraph = nx.DiGraph()
    node_1_name = generate_unique_name_for_type("1*")
    node_C_name = generate_unique_name_for_type("C*")
    node_R_name = generate_unique_name_for_type("R*")
    subgraph.add_node(node_1_name, type="1*")
    subgraph.add_node(node_C_name, type="C*")
    subgraph.add_node(node_R_name, type="R*")
    subgraph.add_edge(node_1_name, node_C_name)
    subgraph.add_edge(node_1_name, node_R_name)
    return subgraph, "1*"

def create_subgraph_for_D():
    variant = random.choice([1, 2, 3])
    if variant == 1:
        return create_subgraph_for_D_variant_1()
    elif variant == 2:
        return create_subgraph_for_D_variant_2()
    else:
        return create_subgraph_for_D_variant_3()


# Замена всех узлов определённого типа их подграфами
def replace_all_nodes_with_subgraphs(graph, node_types_to_replace, create_subgraph_functions):
    """
    Заменяет все узлы указанных типов в графе на подграфы, пока они не будут полностью заменены.

    :param graph: исходный граф.
    :param node_types_to_replace: список типов узлов для замены.
    :param create_subgraph_functions: словарь, где ключ — тип узла, значение — функция создания подграфа.
    """
    replaced = True  # Флаг для повторной проверки
    while replaced:
        replaced = False
        for node_type in node_types_to_replace:
            nodes_to_replace = [node for node, data in graph.nodes(data=True) if data["type"] == node_type]
            if nodes_to_replace:
                replaced = True  # Если есть узлы для замены, продолжаем цикл
            for node_to_replace in nodes_to_replace:
                print(f"Заменяем узел ({node_type}): {node_to_replace}")
                incoming_edges = list(graph.in_edges(node_to_replace, data=True))
                outgoing_edges = list(graph.out_edges(node_to_replace, data=True))
                graph.remove_node(node_to_replace)

                # Создаём подграф и добавляем его в граф
                subgraph, connection_node_type = create_subgraph_functions[node_type]()
                graph.add_nodes_from(subgraph.nodes(data=True))
                graph.add_edges_from(subgraph.edges(data=True))

                # Подключаем рёбра
                if connection_node_type:
                    connection_nodes = [
                        node for node, data in subgraph.nodes(data=True) if data["type"] == connection_node_type
                    ]
                else:
                    connection_nodes = []

                if connection_nodes:
                    connection_node = connection_nodes[0]
                    for src, _, edge_data in incoming_edges:
                        graph.add_edge(src, connection_node, **edge_data)
                    for _, dst, edge_data in outgoing_edges:
                        graph.add_edge(connection_node, dst, **edge_data)
                else:
                    # Определяем входные и выходные узлы подграфа
                    subgraph_nodes = list(subgraph.nodes())
                    incoming_targets = [n for n in subgraph_nodes if subgraph.in_degree(n) == 0]
                    outgoing_sources = [n for n in subgraph_nodes if subgraph.out_degree(n) == 0]

                    # Подключаем рёбра
                    for src, _, edge_data in incoming_edges:
                        for target in incoming_targets:
                            graph.add_edge(src, target, **edge_data)
                    for _, dst, edge_data in outgoing_edges:
                        for source in outgoing_sources:
                            graph.add_edge(source, dst, **edge_data)


# Визуализация графа
def visualize_with_ports_and_subgraphs():
    graph = nx.DiGraph()
    node_G = generate_unique_name_for_type("G*")
    node_Y_1 = generate_unique_name_for_type("Y*")
    node_SE = generate_unique_name_for_type("SE*")
    graph.add_node(node_G, type="G*")
    graph.add_node(node_Y_1, type="Y*")
    graph.add_node(node_SE, type="SE*")
    graph.add_edge(node_G, node_Y_1)
    graph.add_edge(node_Y_1, node_SE)

    print(f"Граф перед заменой: {graph.nodes(data=True)}\n")

    # Определяем словарь с функциями создания подграфов
    create_subgraph_functions = {
        "Y*": create_subgraph_for_Y,
        "G*": create_subgraph_for_G,
        "X*": create_subgraph_for_X,
        "Z*": create_subgraph_for_Z,
        "D*": create_subgraph_for_D,
    }

    # Выполняем замену
    replace_all_nodes_with_subgraphs(graph, ["Y*", "G*", "X*", "Z*", "D*"], create_subgraph_functions)

    print("\nГраф после замен:")
    print_graph_to_file(graph, "graph_output.txt")

    # Визуализация
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edge_color="gray", arrows=True)
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: f"{node}\n{data.get('type', 'N/A')}" for node, data in graph.nodes(data=True)},
        font_size=10,
    )
    plt.title("Граф с заменами", fontsize=14)
    plt.axis("off")
    plt.show()

    return graph


def print_graph_to_file(graph, filename):
    with open(filename, "w") as file:
        file.write("Узлы:\n")
        for node, data in graph.nodes(data=True):
            file.write(f"  {node}: {data}\n")
        file.write("\nРёбра:\n")
        for src, dst, data in graph.edges(data=True):
            file.write(f"  {src} -> {dst} {data}\n")


