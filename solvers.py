import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Symbolic impedance/admittance matrices
# ----------------------------------------------------------------------

def symbolic_ZY(circ, s: sp.Symbol):
    """
    Compute symbolic impedance (Zb) and admittance (Yb) diagonal matrices.
    """
    Zlist = []
    Ylist = []
    for br in circ.branches:
        Z = br.impedance(s)
        Zlist.append(Z)
        Ylist.append(sp.Integer(0) if Z == sp.oo else sp.simplify(1 / Z))

    Zb = sp.diag(*[(sp.Integer(0) if z == sp.oo else z) for z in Zlist])
    Yb = sp.diag(*Ylist)
    return Zb, Yb, Zlist, Ylist


# ----------------------------------------------------------------------
# Node-based symbolic solver
# ----------------------------------------------------------------------

def solve_node_symbolic(circ, s: sp.Symbol):
    """
    Node-voltage based symbolic solver.
    Supports ideal voltage sources connected to the reference node only.
    Treats independent voltage/current sources as step inputs (value/s) in Laplace domain.
    Returns:
        dict with {'Vn', 'Ib', 'Vb', 'names', 'ref_idx', 'Q'}
    """
    Ared, names, ref_idx = circ.reduced_incidence()
    Zb, Yb, Zlist, Ylist = symbolic_ZY(circ, s)

    # Detect known voltages from ideal V sources to ground
    # Treat V(s) = V0 / s  (step input)
    known_V = {}
    for br in circ.branches:
        if br.btype == 'V':
            if br.n2 == circ.ref_node and br.n1 != circ.ref_node:
                known_V[br.n1] = sp.sympify(br.value) / s
            elif br.n1 == circ.ref_node and br.n2 != circ.ref_node:
                known_V[br.n2] = -sp.sympify(br.value) / s

    unk_nodes = [n for n in names if n != circ.ref_node and n not in known_V]
    unk_idx_map = {n: i for i, n in enumerate(unk_nodes)}

    Vsymbols = sp.symbols(f"Vn0:{len(unk_nodes)}") if unk_nodes else ()
    Vsym_map = {unk_nodes[i]: Vsymbols[i] for i in range(len(unk_nodes))}

    eqs = []
    for u in unk_nodes:
        expr = sp.Integer(0)
        for i, br in enumerate(circ.branches):
            Y = Ylist[i]
            if br.n1 == u or br.n2 == u:
                other = br.n2 if br.n1 == u else br.n1
                Vu = _node_voltage(u, Vsym_map, known_V, circ.ref_node)
                Vother = _node_voltage(other, Vsym_map, known_V, circ.ref_node)
                if Y != sp.Integer(0):
                    expr += sp.simplify(Y * (Vu - Vother))
                if br.btype == 'I':
                    # Treat current source as step in Laplace: I(s) = I0 / s
                    Ival = sp.sympify(br.value) / s
                    expr += Ival if br.n1 == u else -Ival
        eqs.append(sp.simplify(expr))

    if unk_nodes:
        A_mat, b_vec = sp.linear_eq_to_matrix(eqs, list(Vsymbols))
        try:
            sol_vec = A_mat.LUsolve(b_vec)
        except Exception:
            sol = sp.solve(eqs, list(Vsymbols), dict=True)
            sol_vec = sp.Matrix([sp.simplify(sol[0][sym]) for sym in Vsymbols]) if sol else sp.zeros(len(Vsymbols), 1)
        full = _assemble_full_voltage_vector(names, circ.ref_node, known_V, Vsym_map, unk_idx_map, sol_vec)
    else:
        full = [_node_voltage(n, {}, known_V, circ.ref_node) for n in names]

    Vb = sp.Matrix([full[names.index(br.n1)] - full[names.index(br.n2)] for br in circ.branches])
    Ib_adm = sp.Matrix([
        sp.Integer(0) if Ylist[i] == sp.Integer(0) else sp.simplify(Ylist[i] * Vb[i])
        for i in range(len(circ.branches))
    ])
    # Current sources contribution treated as I0/s
    Isrc = sp.Matrix([sp.sympify(br.value) / s if br.btype == 'I' else 0 for br in circ.branches])
    Ib = sp.simplify(Ib_adm + Isrc)

    return {'Vn': sp.Matrix(full), 'Ib': Ib, 'Vb': Vb, 'names': names, 'ref_idx': ref_idx, 'Q': Ared}


# ----------------------------------------------------------------------
# Loop-based symbolic solver
# ----------------------------------------------------------------------

def solve_loop_symbolic(circ, s: sp.Symbol):
    """
    Loop-current based symbolic solver.
    Treats independent voltage sources as step inputs (value/s).
    Returns:
        dict with {'B', 'I_l', 'I_b', 'V_b'}
    """
    B, tree_edges, chords = circ.fundamental_tieset()
    if isinstance(B, sp.Matrix) and B.rows == 0:
        raise RuntimeError("No independent loops found.")
    Zb, Yb, Zlist, Ylist = symbolic_ZY(circ, s)

    # Vs entries now use step representation: V0 / s
    Vs = sp.Matrix([_voltage_source_value(circ, br, s) for br in circ.branches])
    Zdiag = [sp.Integer(0) if z == sp.oo else z for z in Zlist]
    Zb_short = sp.diag(*Zdiag)

    BZBt = sp.simplify(B * Zb_short * B.T)
    BV = sp.simplify(B * Vs)
    Il_syms = sp.symbols(f"Il0:{B.rows}")
    Il_vec = sp.Matrix(Il_syms)

    sol = sp.solve(BZBt * Il_vec - BV, list(Il_vec), dict=True)
    Il = sp.Matrix([sp.simplify(sol[0][sym]) for sym in Il_syms]) if sol else sp.zeros(B.rows, 1)

    Ib = sp.simplify(B.T * Il)
    Vb = sp.simplify(Zb_short * Ib + Vs)

    return {'B': B, 'I_l': Il, 'I_b': Ib, 'V_b': Vb}


# ----------------------------------------------------------------------
# Inverse Laplace Transform and plotting
# ----------------------------------------------------------------------

def inverse_and_plot(expressions, s, t, label_prefix, ylabel):
    """
    Performs inverse Laplace transform for a list of symbolic expressions.
    Returns time-domain numeric data for plotting.
    """
    tv = np.linspace(0, 0.01, 1000)
    results = {}
    for i, expr in enumerate(expressions):
        inv, y = _safe_inverse(expr, s, t, tv)
        if inv is not None:
            results[f"{label_prefix}{i}"] = y
            print(f"Inverse Laplace {label_prefix}{i} ->")
            sp.pprint(inv)

    if results:
        plt.figure(figsize=(8, 4))
        for k, v in results.items():
            plt.plot(tv, v, label=k)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} (time domain)")
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print(f"No {ylabel.lower()} signals to plot.")

    return results


# ----------------------------------------------------------------------
# Internal helper functions
# ----------------------------------------------------------------------

def _node_voltage(node, Vsym_map, known_V, ref_node):
    if node in Vsym_map:
        return Vsym_map[node]
    if node in known_V:
        return known_V[node]
    if node == ref_node:
        return sp.Integer(0)
    return sp.Integer(0)

def _assemble_full_voltage_vector(names, ref_node, known_V, Vsym_map, unk_idx_map, sol_vec):
    full = []
    for n in names:
        if n == ref_node:
            full.append(sp.Integer(0))
        elif n in known_V:
            full.append(sp.simplify(known_V[n]))
        elif n in Vsym_map:
            full.append(sp.simplify(sol_vec[unk_idx_map[n]]))
        else:
            full.append(sp.Integer(0))
    return full

def _voltage_source_value(circ, br, s: sp.Symbol):
    """
    Return Laplace-domain voltage for a branch:
      - If branch is V and one terminal is reference, return ±(value/s)
      - Otherwise return value/s for general V source (user warned elsewhere)
      - Non-voltage branches return 0
    """
    if br.btype == 'V':
        if br.n2 == circ.ref_node:
            return sp.sympify(br.value) / s
        elif br.n1 == circ.ref_node:
            return -sp.sympify(br.value) / s
        else:
            return sp.sympify(br.value) / s
    return sp.Integer(0)

def _safe_inverse(expr, s, t, tv):
    try:
        inv = sp.inverse_laplace_transform(sp.simplify(expr), s, t)
        inv = sp.simplify(inv)
        f = sp.lambdify(t, inv, modules=['numpy', {'Heaviside': lambda x: np.heaviside(x, 1)}])
        y = f(tv)
        y = np.array(y, dtype=float)
        if y.ndim == 0 or y.size == 1:
            y = np.ones_like(tv) * float(y)
        elif y.shape != tv.shape:
            try:
                y = y.reshape(tv.shape)
            except Exception:
                y = np.ones_like(tv) * float(np.mean(y))
        return inv, y
    except Exception:
        return None, None
