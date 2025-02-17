import networkx as nx
import random
import os
import argparse
import numpy


def generate_random_directed_graph(num_nodes, edge_prob, DAG=True):
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if DAG:
                if i < j and random.random() < edge_prob:
                    G.add_edge(i, j)
            else:
                if i != j and random.random() < edge_prob:
                    G.add_edge(i, j)
    return G


def find_paths(G, source, target, max_paths=10):
    paths = list(nx.all_simple_paths(G, source, target))
    return paths[:max_paths]


def find_random_cut(G, source, target):
    # Create a copy of G with unit capacities on all edges
    G_cap = G.copy()
    for u, v in G_cap.edges():
        G_cap[u][v]['capacity'] = 1  # Assign unit capacity to avoid infinite flow

    cut_value, (reachable, non_reachable) = nx.minimum_cut(G_cap, source, target)
    remaining_nodes = sorted(set(G.nodes()) - {target} - non_reachable)
    num_additional = random.randint(1, len(remaining_nodes) // 10) if remaining_nodes else 0
    additional_nodes = set(random.sample(remaining_nodes, num_additional))

    return non_reachable | additional_nodes


def create_dataset(G, num_nodes, max_paths=10, test_prob=0.5):
    train_dataset = []
    test_dataset = []

    for s in range(num_nodes):
        for t in range(s + 1, num_nodes):
            if nx.has_path(G, s, t):
                paths = find_paths(G, s, t, max_paths)
                cut = find_random_cut(G, s, t)
                paths_str = " # ".join([" ".join(map(str, path[1:])) for path in paths])
                cut_str = " ".join(map(str, cut))
                entry = f"{s} {t} ! {paths_str} % "

                if random.random() < test_prob:
                    entry = entry + "\n"
                    test_dataset.append(entry)
                else:
                    entry = entry + f"{cut_str}\n"
                    train_dataset.append(entry)

    return train_dataset, test_dataset


def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        file.writelines(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random graph and find (s, t) cuts.')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')
    parser.add_argument('--edge_prob', type=float, default=0.1, help='Edge creation probability')
    parser.add_argument('--DAG', type=bool, default=True, help='If true, generate a DAG')
    parser.add_argument('--max_paths', type=int, default=10, help='Max paths to store per (s,t) pair')
    parser.add_argument('--test_prob', type=float, default=0.5, help='Probability of assigning a cut to test data')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    edge_prob = args.edge_prob
    DAG = args.DAG
    max_paths = args.max_paths
    test_prob = args.test_prob

    random_digraph = generate_random_directed_graph(num_nodes, edge_prob, DAG)
    train_dataset, test_dataset = create_dataset(random_digraph, num_nodes, max_paths, test_prob)

    folder_name = f"{num_nodes}_cut"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    write_dataset(train_dataset,  os.path.join(folder_name, f"train_{max_paths}.txt"))
    write_dataset(test_dataset, os.path.join(folder_name, f"test.txt"))
    nx.write_graphml(random_digraph, os.path.join(folder_name, "graph.graphml"))
