from dataclasses import dataclass
import sympy as sp

@dataclass
class Branch:
    name: str
    btype: str  # 'R','L','C','V','I'
    n1: str
    n2: str
    value: float
    index: int

    def impedance(self, s: sp.Symbol):
        if self.btype == 'R':
            return sp.sympify(self.value)
        if self.btype == 'L':
            return s * sp.sympify(self.value)
        if self.btype == 'C':
            return 1 / (s * sp.sympify(self.value))
        if self.btype in ('V', 'I'):
            return sp.oo
        raise ValueError(f'Unknown branch type {self.btype}')

    def admittance(self, s: sp.Symbol):
        Z = self.impedance(s)
        if Z == sp.oo:
            return sp.Integer(0)
        return sp.simplify(1 / Z)
