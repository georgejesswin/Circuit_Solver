import sympy as sp
from circuit import Circuit

def parse_netlist(path: str) -> Circuit:
    circ = Circuit(ref_node='0')
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', ';')):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            t, name, n1, n2 = parts[:4]
            val = parts[4] if len(parts) > 4 else '0'
            try:
                fval = float(val)
            except Exception:
                try:
                    fval = float(sp.N(sp.sympify(val)))
                except Exception:
                    fval = 0.0
            btype = t.upper()[0]
            circ.add_branch(btype, name, n1, n2, fval)
    return circ
