import sympy as sp
import matplotlib.pyplot as plt
import networkx as nx

def print_matrix_with_indices(M, row_names, col_names, title):
    print(f"\n{title} (rows x cols = {M.rows} x {M.cols}):")
    if M.rows == 0 or M.cols == 0:
        print('  <empty>')
        return
    header = '     ' + ' '.join([f"c{j:>6}" for j in range(M.cols)])
    print(header)
    for i in range(M.rows):
        rowstr = f"r{i:>3}:"
        for j in range(M.cols):
            rowstr += f" {str(sp.simplify(M[i, j])):>6}"
        print(rowstr)

def draw_graph(circ):
    names = sorted(circ.nodes.keys(), key=lambda x: (x != '0', x))
    pos = nx.spring_layout(circ.G, seed=2)
    plt.figure(figsize=(6, 4))
    nx.draw_networkx_nodes(circ.G, pos, node_size=600)
    nx.draw_networkx_labels(circ.G, pos)

    pair_map = {}
    for u, v, key, data in circ.G.edges(keys=True, data=True):
        pair = tuple(sorted((u, v)))
        pair_map.setdefault(pair, []).append((u, v, key, data))

    for pair, edges in pair_map.items():
        m = len(edges)
        for k, (u, v, key, data) in enumerate(edges):
            rad = (k - (m - 1) / 2) * 0.25 if m > 1 else 0.0
            nx.draw_networkx_edges(circ.G, pos, edgelist=[(u, v)], connectionstyle=f'arc3, rad={rad}')
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = y2 - y1, -(x2 - x1)
            norm = (dx**2 + dy**2)**0.5 or 1
            off = 0.02 * (k - (m - 1) / 2)
            plt.text(xm + off * dx / norm, ym + off * dy / norm,
                     f"b{data['object'].index}", fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    plt.title('Circuit (branch indices shown)')
    plt.axis('off')
    plt.show()
