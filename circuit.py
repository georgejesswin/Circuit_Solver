import networkx as nx
import sympy as sp
from typing import List, Dict, Tuple
from branch import Branch
from matrices import build_incidence_matrix, build_reduced_incidence, build_fundamental_tieset
from solvers import solve_node_symbolic, solve_loop_symbolic
from utils import print_matrix_with_indices, draw_graph

class Circuit:
    def __init__(self, ref_node: str = '0'):
        self.branches: List[Branch] = []
        self.nodes: Dict[str, int] = {}
        self.ref_node = str(ref_node)
        self.G = nx.MultiGraph()

    def add_branch(self, btype, name, n1, n2, value):
        n1, n2 = str(n1), str(n2)
        for n in (n1, n2):
            if n not in self.nodes:
                self.nodes[n] = len(self.nodes)
                self.G.add_node(n)
        idx = len(self.branches)
        br = Branch(name=name, btype=btype, n1=n1, n2=n2, value=float(value), index=idx)
        self.branches.append(br)
        self.G.add_edge(n1, n2, key=idx, object=br)
        return br

    # Matrix generation
    def incidence_matrix(self):
        return build_incidence_matrix(self)

    def reduced_incidence(self):
        return build_reduced_incidence(self)

    def fundamental_tieset(self):
        return build_fundamental_tieset(self)

    # Symbolic solvers
    def solve_node_symbolic(self, s: sp.Symbol):
        return solve_node_symbolic(self, s)

    def solve_loop_symbolic(self, s: sp.Symbol):
        return solve_loop_symbolic(self, s)

    # Utilities
    def print_indices(self):
        names = sorted(self.nodes.keys(), key=lambda x: (x != '0', x))
        print('\nNode indices:')
        for i, n in enumerate(names):
            print(f"  idx {i}: '{n}'")
        print('\nBranch indices:')
        for br in self.branches:
            print(f"  idx {br.index}: '{br.name}'  {br.n1}->{br.n2} ({br.btype}, val={br.value})")

    def print_matrix(self, M, row_names, col_names, title):
        print_matrix_with_indices(M, row_names, col_names, title)

    def draw_graph(self):
        draw_graph(self)
