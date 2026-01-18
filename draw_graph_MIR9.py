import networkx as nx
import matplotlib.pyplot as plt

def get_next_state_binary(current_state_str):
    """
    calculates next state
    """
    x = [int(b) for b in current_state_str]
    x0, x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4], x[5]

    # Predictor Functions
    # f0: x2
    nx0 = x2
    
    # f1: ~x3 and x5 or x3 <->  (x3 or x5)
    nx1 = 1 if (x3 == 1 or x5 == 1) else 0
    
    # f2: ~x4 and ~x1
    nx2 = 1 if (x4 == 0 and x1 == 0) else 0
    
    # f3: ~x4 and ~x0
    nx3 = 1 if (x4 == 0 and x0 == 0) else 0
    
    # f4: ~x3 and ~x0
    nx4 = 1 if (x3 == 0 and x0 == 0) else 0
    
    # f5: ~x4 and ~x0
    nx5 = 1 if (x4 == 0 and x0 == 0) else 0

    return f"{nx0}{nx1}{nx2}{nx3}{nx4}{nx5}"

# Creating an empty graph
G = nx.DiGraph()

# Generating 2^6=64 states
for i in range(64):
    current_state = format(i, '06b') 
    next_state = get_next_state_binary(current_state)
    
    G.add_edge(current_state, next_state)

plt.figure(figsize=(14, 12))

attractors = [node for node in G.nodes() if list(G.successors(node))[0] == node]

# setting yellow for attractor, blue for other node
node_colors = ['yellow' if node in attractors else '#cceeff' for node in G.nodes()]

node_sizes = [1500 for _ in G.nodes()]

pos = nx.spring_layout(G, k=0.8, iterations=50)

# -- Drawing --
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')

nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=15, arrowstyle='-|>',width=1.0,min_source_margin=25,min_target_margin=25)

nx.draw_networkx_labels(G, pos, font_size=9, font_family='monospace', font_weight='bold')

plt.title("State Transition Graph for MIR9", fontsize=16)
plt.axis('off')

plt.show()