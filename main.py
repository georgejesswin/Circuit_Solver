import sys
import sympy as sp
import numpy as np
from parser import parse_netlist
from solvers import inverse_and_plot

def run(netlist_path: str):
    circ = parse_netlist(netlist_path)
    circ.print_indices()
    circ.draw_graph()  # <-- Graph of the circuit topology

    # Matrices
    Ared, names_all, ref_idx = circ.reduced_incidence()
    B, tree_edges, chords = circ.fundamental_tieset()
    circ.print_matrix(Ared, [], [f'b{j}' for j in range(len(circ.branches))],
                      'Reduced incidence Q (rows non-ref nodes, cols branches)')
    circ.print_matrix(B, [f'l{i}' for i in range(B.rows)] if B.rows > 0 else [],
                      [f'b{j}' for j in range(len(circ.branches))],
                      'Fundamental tie-set B (loops x branches)')

    # Symbolic solve
    s, t = sp.symbols('s t')
    vs_between_nonref = any(
        br.btype == 'V' and br.n1 != circ.ref_node and br.n2 != circ.ref_node
        for br in circ.branches
    )

    try:
        if not vs_between_nonref:
            result = circ.solve_node_symbolic(s)
        else:
            result = circ.solve_loop_symbolic(s)
    except Exception as e:
        print("Symbolic solver error:", e)
        return

    Vn_sym = list(result.get('Vn', []))
    Ib_sym = list(result.get('Ib', []))
    names_used = result.get('names', [])

    # Print symbolic (Laplace domain) results
    print("\nLaplace-domain node voltages:")
    for i, expr in enumerate(Vn_sym):
        node_name = names_used[i] if i < len(names_used) else f"node{i}"
        print(f" n{i} (\"{node_name}\") = {sp.simplify(expr)}")

    print("\nLaplace-domain branch currents:")
    for i, expr in enumerate(Ib_sym):
        br = circ.branches[i]
        print(f" b{i} (\"{br.name}\" {br.n1}->{br.n2}) = {sp.simplify(expr)}")

    # ---- Time-domain plots ----
    print("\n=== Inverse Laplace and time-domain plotting ===")
    if Vn_sym:
        inverse_and_plot(Vn_sym, s, t, "n", "Voltage (V)")
    if Ib_sym:
        inverse_and_plot(Ib_sym, s, t, "b", "Current (A)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m graph_circuit_sim.main <netlist.cir>")
        sys.exit(1)
    run(sys.argv[1])
