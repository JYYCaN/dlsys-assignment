from tkinter import font
import networkx as nx
import matplotlib.pyplot as plt
from utils import find_topo_order
def plot_conputation_graph(node_list, node_size=1200):
    G = nx.DiGraph()
    edge_list = []
    nodes = find_topo_order(node_list)
    G = nx.DiGraph()
    for node in nodes:
        edge_list.extend([(input.name, node.name) for input in node.inputs])
    G.add_edges_from(edge_list)
    G.add_nodes_from([node.name for node in nodes])
    pos = nx.layout.circular_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, font_size=8)
    
    plt.show()
    edge_list = []
    return node_list
