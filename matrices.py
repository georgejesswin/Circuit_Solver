import sympy as sp
import networkx as nx

def build_incidence_matrix(circ):
    names = sorted(circ.nodes.keys(), key=lambda x: (x != '0', x))
    n = len(names)
    b = len(circ.branches)
    A = sp.zeros(n, b)
    for j, br in enumerate(circ.branches):
        i1 = names.index(br.n1)
        i2 = names.index(br.n2)
        A[i1, j], A[i2, j] = 1, -1
    return A, names

def build_reduced_incidence(circ):
    A, names = build_incidence_matrix(circ)
    ridx = names.index(circ.ref_node) if circ.ref_node in names else 0
    Ared = A[[i for i in range(A.rows) if i != ridx], :]
    return Ared, names, ridx

def build_fundamental_tieset(circ):
    und = nx.Graph()
    for br in circ.branches:
        und.add_edge(br.n1, br.n2, idx=br.index)
    if und.number_of_nodes() == 0:
        return sp.Matrix([]), set(), []
    T = nx.minimum_spanning_tree(und)
    tree_edge_indices = {und[u][v]['idx'] for u, v in T.edges()}
    chords = [i for i in range(len(circ.branches)) if i not in tree_edge_indices]
    loops = []
    for chord_idx in chords:
        chord = circ.branches[chord_idx]
        try:
            path = nx.shortest_path(T, chord.n1, chord.n2)
        except nx.NetworkXNoPath:
            path = []
        row = [0] * len(circ.branches)
        for a, b in zip(path[:-1], path[1:]):
            idx = und[a][b]['idx']
            sign = 1 if (circ.branches[idx].n1 == a and circ.branches[idx].n2 == b) else -1
            row[idx] = sign
        row[chord_idx] = 1
        loops.append(row)
    return sp.Matrix(loops), tree_edge_indices, chords
