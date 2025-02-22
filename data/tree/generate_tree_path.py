import networkx as nx
import random
import os
import argparse
import numpy as np


def generate_random_rooted_tree(num_nodes):
    G = nx.DiGraph()
    G.add_node(0)  # Root node

    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)  # Ensure tree structure
        G.add_edge(parent, i)

    return G


def get_root_to_leaf_paths(G):
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    leaves = [node for node in G.nodes if G.out_degree(node) == 0]

    paths = []
    for leaf in leaves:
        path = []
        current = leaf
        while current in G:
            path.append(current)
            preds = list(G.predecessors(current))
            if preds:
                current = preds[0]
            else:
                break
        paths.append(list(reversed(path)))

    return paths


def create_dataset(G):
    train_set = []
    test_set = get_root_to_leaf_paths(G)  # Root-to-leaf paths only in test

    covered_edges = set()
    all_edges = set(G.edges())

    while covered_edges != all_edges:
        path = random.choice(test_set)  # Pick a root-to-leaf path
        if len(path) > 2:
            subpath_length = random.randint(2, len(path) - 1)
        else:
            subpath_length = 2  # Default to the minimum valid length

        subpath_start = random.randint(0, len(path) - subpath_length)
        subpath = path[subpath_start: subpath_start + subpath_length]

        train_set.append(subpath)

        for i in range(len(subpath) - 1):
            covered_edges.add((subpath[i], subpath[i + 1]))

    return train_set, test_set


def format_data(data):
    return f"{data[0]} {data[-1]} " + ' '.join(str(num) for num in data) + '\n'


def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random rooted tree and datasets.')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the tree')
    args = parser.parse_args()

    num_nodes = args.num_nodes
    rooted_tree = generate_random_rooted_tree(num_nodes)
    train_set, test_set = create_dataset(rooted_tree)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree')
    os.makedirs(folder_name, exist_ok=True)


    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/train_20.txt'))
    write_dataset(test_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/test.txt'))
    nx.write_graphml(rooted_tree, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/path_graph.graphml'))

    print(f"Dataset generated in {folder_name}/")
