import networkx as nx
import matplotlib.pyplot as plt

def build_graph():
    G = nx.DiGraph()
    G.add_node(0, role="supplier")
    G.add_node(1, role="manufacturer")
    G.add_node(2, role="retailer")
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    return G

G = build_graph()

pos = {0: (0, 0), 1: (1, 0), 2: (2, 0)}  # fixed layout (nice for paper)
labels = {n: f"{n}\n{G.nodes[n]['role']}" for n in G.nodes()}

plt.figure(figsize=(6, 2))
nx.draw_networkx_nodes(G, pos, node_size=1800)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=2)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

plt.axis("off")
plt.tight_layout()
plt.savefig("supply_chain_graph.png", dpi=300)
print("Wrote supply_chain_graph.png")
