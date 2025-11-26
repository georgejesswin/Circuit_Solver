# Circuit Solver – EE-204 Cicruit Theory Course Project

A lightweight Python-based **circuit solver and schematic editor** implementing  
**Modified Nodal Analysis (MNA)** with both **numerical** and **symbolic** capabilities.  
It supports SPICE-style netlists, GUI-based circuit construction, automatic matrix
assembly, and solution of DC / Laplace-domain circuits.

---

## 🚀 Features

- **Interactive GUI Editor** (Tkinter) for drawing circuits  
- Place R, L, C, V/I sources and connect nodes  
- SPICE-like netlist support (`example.cir`)  
- **Modified Nodal Analysis (MNA)** formulation  
- **Symbolic solving** using SymPy  
- **Numerical solving** using NumPy  
- Automatic matrix construction (G, B, C, D)  
- Node voltages, branch currents, time-domain responses  
- Waveform plotting & result visualization  

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| **components.py** | Component classes (R, L, C, V/I sources) with stamping helpers. |
| **circuit.py** | Circuit model: nodes, components, validation, representation. |
| **branch.py** | Branch representation between nodes and components. |
| **matrices.py** | Builds MNA matrices (G(s), B, BT) and RHS vectors. |
| **solvers.py** | Numerical & symbolic solvers; diagnostics and post-processing. |
| **parser.py** | SPICE-like netlist parser → Circuit object. |
| **example.cir** | Example netlist demonstrating supported syntax. |
| **editor.py** | Tkinter-based schematic editor: draw, edit, simulate circuits. |
| **main.py** | CLI entry point for netlist-based circuit solving. |
| **main_gui.py** | Minimal launcher for the GUI editor. |
| **utils.py** | Unit parsing, formatting helpers, numeric utilities. |

---

## 🧠 Solution Approach: Modified Nodal Analysis (MNA)

The solver constructs and solves:

A_mna(s) x(s) = b_mna(s)

Where:
- Unknowns = node voltages + currents through independent voltage sources  
- Matrix structure:

A_mna(s) =
[ G(s)   B  ]
[ B^T   0  ]

Supports:
- Laplace-domain modeling of R, L, C  
- Symbolic inversion and time-domain recovery  
- Numerical simulation  
- Detection of singular matrices and floating nodes  

---

## 🖥️ Typical Workflow

1. Draw circuit in GUI **or** load `.cir` file  
2. Parse with `parser.py`  
3. Assemble matrices via `matrices.py`  
4. Solve using `solvers.py`  
5. Display voltages, currents, plots  

---

## ▶️ Running the Project

### CLI (netlist)
```bash
python main.py example.cir
```

### GUI
```bash
python main_gui.py
```

---

## 👥 Project Group 

- **Jesswin George (240102042)**  
- **Achuyth A. (240102003)**  
- **Mathew James (240102061)**  
