# solvers.py  — MNA-only symbolic circuit solver (clean, robust)
# Always uses Modified Nodal Analysis (MNA) for consistency.

from typing import List, Tuple, Dict, Any, Union, Optional
import logging

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# configure logging for module users
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

__all__ = [
    "symbolic_ZY",
    "solve_node_symbolic",  # kept as wrapper for compatibility (delegates to MNA)
    "solve_mna_symbolic",
    "solve_loop_symbolic",
    "inverse_and_plot",
]

# ----------------- small helpers -------------------------------------------
def _is_zero(expr: sp.Expr) -> bool:
    try:
        return sp.simplify(expr) == sp.S.Zero
    except Exception:
        return False


def _to_sym(expr):
    return sp.sympify(expr) if expr is not None else sp.S.Zero


# ---------------------------------------------------------------------------
# Symbolic Z/Y extraction
# ---------------------------------------------------------------------------
def symbolic_ZY(circ, s: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix, List[sp.Expr], List[sp.Expr]]:
    """
    Returns (Zb, Yb, Zlist, Ylist)
    - Zb: diagonal matrix of branch impedances (with sp.S.Zero for ideal short)
    - Yb: diagonal matrix of branch admittances (with sp.S.Zero for open/undefined)
    - Zlist/Ylist: lists for each branch (branch order = circ.branches)
    """
    Zlist: List[sp.Expr] = []
    Ylist: List[sp.Expr] = []

    for br in circ.branches:
        Z = None
        try:
            Z = br.impedance(s)
        except Exception:
            # Ideal sources or unspecified impedance -> treat specially
            if getattr(br, "btype", None) == "V":
                Z = sp.S.Zero  # ideal short for impedance representation (handled elsewhere)
            else:
                Z = sp.oo

        try:
            Z = sp.simplify(Z)
        except Exception:
            Z = sp.sympify(Z)

        if Z == sp.oo:
            Y = sp.S.Zero
        else:
            if Z == sp.S.Zero:
                Y = sp.oo
            else:
                try:
                    Y = sp.simplify(1 / Z)
                except Exception:
                    Y = sp.S.Zero

        Zlist.append(Z)
        Ylist.append(Y)

    Zdiag = [(sp.S.Zero if z == sp.oo else z) for z in Zlist]
    Zb = sp.diag(*Zdiag) if Zdiag else sp.zeros(0)
    Yb = sp.diag(*Ylist) if Ylist else sp.zeros(0)

    return Zb, Yb, Zlist, Ylist


# ---------------------------------------------------------------------------
# MNA solver (the single canonical solver used by this module)
# ---------------------------------------------------------------------------
def solve_mna_symbolic(circ, s: sp.Symbol) -> Dict[str, Any]:
    """
    Always use Modified Nodal Analysis (MNA).
    Returns dictionary with keys similar to earlier interface:
      - 'Vn' : full node voltage vector (including reference at its proper index)
      - 'I_branch' : branch currents (matching circ.branches order)
      - 'V_branch' : branch voltages (V(n1) - V(n2))
      - 'names' : list of node names (including reference)
      - 'ref_idx' : index of reference node within names
      - 'A_mna', 'b_mna' : final MNA matrix and RHS (useful for debugging)
    """
    Ared, names, ref_idx = circ.reduced_incidence()
    node_list = [n for n in names if n != circ.ref_node]
    N = len(node_list)

    Zb, Yb, Zlist, Ylist = symbolic_ZY(circ, s)

    # Collect voltage sources for MNA augmentation
    v_sources = [(i, br) for i, br in enumerate(circ.branches) if getattr(br, "btype", None) == "V"]
    M = len(v_sources)

    G = sp.zeros(N, N)
    Ivec = sp.zeros(N, 1)

    def node_idx(n):
        return None if n == circ.ref_node else node_list.index(n)

    # Assemble G and Ivec (for passive admittances and current sources)
    for i, br in enumerate(circ.branches):
        Y = Ylist[i]
        n1 = br.n1
        n2 = br.n2

        if getattr(br, "btype", None) == "I":
            Ival = sp.simplify(sp.sympify(br.value) / s)
            idx1 = node_idx(n1)
            idx2 = node_idx(n2)
            if idx1 is not None:
                Ivec[idx1, 0] -= Ival
            if idx2 is not None:
                Ivec[idx2, 0] += Ival

        if not _is_zero(Y) and getattr(br, "btype", None) != "V":
            idx1 = node_idx(n1)
            idx2 = node_idx(n2)
            if idx1 is not None:
                G[idx1, idx1] += Y
            if idx2 is not None:
                G[idx2, idx2] += Y
            if idx1 is not None and idx2 is not None:
                G[idx1, idx2] -= Y
                G[idx2, idx1] -= Y

    # B and E for voltage sources
    B = sp.zeros(N, M)
    E = sp.zeros(M, 1)
    for j, (branch_idx, br) in enumerate(v_sources):
        n1 = br.n1
        n2 = br.n2
        idx1 = node_idx(n1)
        idx2 = node_idx(n2)
        if idx1 is not None:
            B[idx1, j] = +1
        if idx2 is not None:
            B[idx2, j] = -1
        E[j, 0] = sp.simplify(sp.sympify(br.value) / s)

    # Build final MNA matrix
    if N == 0 and M == 0:
        A_mna = sp.zeros(0)
        b_mna = sp.zeros(0, 1)
        V_unknown = sp.zeros(0, 1)
        I_v = sp.zeros(0, 1)
    else:
        if M == 0:
            A_mna = G
            b_mna = Ivec
        elif N == 0:
            A_mna = sp.zeros(M, M)
            b_mna = E
        else:
            top = sp.Matrix.hstack(G, B)
            bottom = sp.Matrix.hstack(B.T, sp.zeros(M, M))
            A_mna = sp.Matrix.vstack(top, bottom)
            b_mna = sp.Matrix.vstack(Ivec, E)

        try:
            sol = A_mna.LUsolve(b_mna)
        except Exception:
            try:
                x_syms = sp.Matrix(sp.symbols(f"x0:{A_mna.shape[1]}"))
                sol_list = sp.solve(A_mna * x_syms - b_mna, list(x_syms), dict=True)
                if sol_list:
                    sol = sp.Matrix([sp.simplify(sol_list[0][sym]) for sym in list(x_syms)])
                else:
                    sol = sp.zeros(A_mna.shape[1], 1)
            except Exception:
                sol = sp.zeros(A_mna.shape[1], 1)

        V_unknown = sol[:N, :] if N > 0 else sp.zeros(0, 1)
        I_v = sol[N:, :] if M > 0 else sp.zeros(0, 1)

    # Build full node voltages list aligned with names
    Vn_full: List[sp.Expr] = []
    for n in names:
        if n == circ.ref_node:
            Vn_full.append(sp.S.Zero)
        else:
            idx = node_list.index(n)
            Vn_full.append(sp.simplify(V_unknown[idx, 0]))

    # Branch voltages and currents
    Vb = sp.Matrix([Vn_full[names.index(br.n1)] - Vn_full[names.index(br.n2)] for br in circ.branches])

    vs_branch_to_iv = {branch_idx: -I_v[j, 0] for j, (branch_idx, br) in enumerate(v_sources)} if M > 0 else {}

    Ib_list: List[sp.Expr] = []
    for i, br in enumerate(circ.branches):
        if getattr(br, "btype", None) == "V":
            Ib_list.append(sp.simplify(vs_branch_to_iv.get(i, sp.S.Zero)))
        elif getattr(br, "btype", None) == "I":
            Ib_list.append(sp.simplify(sp.sympify(br.value) / s))
        else:
            Y = Ylist[i]
            if _is_zero(Y) or Y == sp.oo:
                Ib_list.append(sp.S.Zero)
            else:
                Ib_list.append(sp.simplify(Y * Vb[i]))
    Ib = sp.Matrix(Ib_list)

    return {
        "Vn": sp.Matrix(Vn_full),
        "I_branch": Ib,
        "V_branch": Vb,
        "names": names,
        "ref_idx": ref_idx,
        "A_mna": A_mna,
        "b_mna": b_mna,
    }


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper: always delegate to MNA (simpler API)
# ---------------------------------------------------------------------------
def solve_node_symbolic(circ, s: sp.Symbol) -> Dict[str, Any]:
    """
    Wrapper kept for compatibility with code that calls solve_node_symbolic.
    Internally this always uses MNA for robust and consistent results.
    """
    logger.debug("solve_node_symbolic: delegating to solve_mna_symbolic (MNA-only mode)")
    out = solve_mna_symbolic(circ, s)

    # For compatibility with older callers expecting 'Ib'/'Vb' keys used by nodal solver,
    # keep those names as aliases.
    return {
        "Vn": out["Vn"],
        "Ib": out["I_branch"],
        "Vb": out["V_branch"],
        "names": out["names"],
        "ref_idx": out["ref_idx"],
        "A_mna": out.get("A_mna"),
        "b_mna": out.get("b_mna"),
    }


# ---------------------------------------------------------------------------
# Loop solver (kept for completeness)
# ---------------------------------------------------------------------------
def _voltage_source_value(circ, br, s: sp.Symbol) -> sp.Expr:
    if getattr(br, "btype", None) == "V":
        return sp.simplify(sp.sympify(br.value) / s)
    return sp.S.Zero


def solve_loop_symbolic(circ, s: sp.Symbol) -> Dict[str, Any]:
    B, tree_edges, chords = circ.fundamental_tieset()
    if not isinstance(B, sp.Matrix) or B.rows == 0:
        raise RuntimeError("No independent loops found for loop analysis.")

    Zb, Yb, Zlist, Ylist = symbolic_ZY(circ, s)

    Vs = sp.Matrix([_voltage_source_value(circ, br, s) for br in circ.branches])

    Zdiag = [sp.S.Zero if z == sp.oo else z for z in Zlist]
    Zb_short = sp.diag(*Zdiag) if Zdiag else sp.zeros(0)

    BZBt = sp.simplify(B * Zb_short * B.T)
    BV = sp.simplify(B * Vs)

    Il_syms = sp.symbols(f"Il0:{B.rows}")
    Il_vec = sp.Matrix(Il_syms)

    try:
        sol_dicts = sp.solve(BZBt * Il_vec - BV, list(Il_vec), dict=True)
    except Exception:
        sol_dicts = []

    if sol_dicts:
        Il = sp.Matrix([sp.simplify(sol_dicts[0][sym]) for sym in Il_syms])
    else:
        try:
            Il = BZBt.LUsolve(BV)
        except Exception:
            Il = sp.zeros(B.rows, 1)

    Ib = sp.simplify(B.T * Il)
    Vb = sp.simplify(Zb_short * Ib + Vs)

    return {"B": B, "I_l": Il, "I_b": Ib, "V_b": Vb}


def inverse_and_plot(
    expressions,
    s,
    t_sym,
    tv=None,
    t_end=0.01,
    n_pts=1000,
    title_prefix="sig",
    ylabel="Signal",
    series_labels=None,
    mode="stacked",
    verbose=True,
):
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # Build list of (expression, label)
    # ============================================================
    items = []

    if isinstance(expressions, sp.Matrix):
        for i in range(expressions.rows):
            label = (
                series_labels[i]
                if series_labels and i < len(series_labels)
                else f"{title_prefix}{i}"
            )
            items.append((expressions[i, 0], label))

    elif isinstance(expressions, (list, tuple)):
        for i, e in enumerate(expressions):
            if isinstance(e, (list, tuple)) and len(e) == 2:
                # explicit label in tuple
                label = (
                    series_labels[i]
                    if series_labels and i < len(series_labels)
                    else str(e[1])
                )
                items.append((sp.simplify(e[0]), label))
            else:
                # auto label or from series_labels
                label = (
                    series_labels[i]
                    if series_labels and i < len(series_labels)
                    else f"{title_prefix}{i}"
                )
                items.append((sp.simplify(e), label))

    else:
        label = (
            series_labels[0]
            if series_labels and len(series_labels) > 0
            else f"{title_prefix}0"
        )
        items = [(sp.simplify(expressions), label)]

    # ============================================================
    # Generate time vector
    # ============================================================
    if tv is None:
        tv = np.linspace(0, t_end, n_pts)
    else:
        tv = np.asarray(tv, float)

    results = {}

    # ============================================================
    # Perform inverse Laplace on each expression
    # ============================================================
    for expr, label in items:
        try:
            inv = sp.simplify(sp.inverse_laplace_transform(expr, s, t_sym))
            f = sp.lambdify(t_sym, inv, ["numpy", {"Heaviside": np.heaviside}])
            y = np.array(f(tv), dtype=float)

            if y.ndim == 0:
                y = np.ones_like(tv) * float(y)
            elif y.shape != tv.shape:
                try:
                    y = y.reshape(tv.shape)
                except:
                    y = np.ones_like(tv) * float(np.mean(y))

            results[label] = y

        except Exception as ex:
            if verbose:
                print(f"Inverse Laplace failed for {label}: {ex}")
            continue

    if len(results) == 0:
        return results

    labels = list(results.keys())
    signals = list(results.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

    # ============================================================
    # Plot modes
    # ============================================================
    if mode == "stacked":
        n = len(signals)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for idx, (label, y) in enumerate(results.items()):
            ax = axes[idx]
            ax.plot(tv, y, color=colors[idx], linewidth=1.8)

            ymin, ymax = np.min(y), np.max(y)
            r = ymax - ymin or 1e-12
            ax.set_ylim(ymin - 0.1*r, ymax + 0.1*r)
            ax.set_ylabel(label, rotation=0, labelpad=40, fontsize=11)

            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{ylabel} – {title_prefix}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    elif mode == "offset":
        plt.figure(figsize=(10, 5))
        peak_ranges = [np.ptp(y) for y in signals]
        base = np.max(peak_ranges) or 1
        offset_step = 0.25 * base

        for idx, (label, y) in enumerate(results.items()):
            offset = idx * offset_step
            plt.plot(tv, y + offset,
                     label=f"{label} (+{offset:.3g})",
                     alpha=0.85, linewidth=1.8, color=colors[idx])

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} – {title_prefix}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif mode == "overlaid":
        plt.figure(figsize=(10, 5))
        for idx, (label, y) in enumerate(results.items()):
            plt.plot(tv, y, label=label, linewidth=1.8, color=colors[idx])

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} – {title_prefix}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("mode must be 'stacked', 'offset', or 'overlaid'")

    return results



