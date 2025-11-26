# editor.py

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, scrolledtext, filedialog
import math
import sympy as sp
import json
import tempfile
import os

from components import ComponentItem, GRID_SIZE, NODE_RADIUS, COLORS

try:
    from parser import parse_netlist
    from solvers import solve_node_symbolic, solve_mna_symbolic, inverse_and_plot
    BACKEND_AVAILABLE = True
except Exception:
    BACKEND_AVAILABLE = False

    def parse_netlist(path):
        raise RuntimeError("parse_netlist not available.")

    def solve_node_symbolic(circ, s):
        raise RuntimeError("solve_node_symbolic not available.")

    def solve_mna_symbolic(circ, s):
        raise RuntimeError("solve_mna_symbolic not available.")

    def inverse_and_plot(*args, **kwargs):
        raise RuntimeError("inverse_and_plot not available.")


class SchematicEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Circuit Solver GUI v2.0")
        self.geometry("1200x800")

        self.nodes = {}
        self.components = []
        self.next_node_idx = 1
        self.comp_counters = {"R": 1, "C": 1, "L": 1, "V": 1, "I": 1}

        self.mode = "select"
        self.temp_start_node = None
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.scale = 1.0

        self._init_union_find()
        self._ground_insert_at = None

        self._init_ui()
        self._bind_events()

    def _init_ui(self):
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        tools_frame = ttk.Frame(main_paned, width=200, relief=tk.RAISED)
        main_paned.add(tools_frame, weight=0)

        ttk.Label(tools_frame, text="Tools", font=("Arial", 10, "bold")).pack(pady=5)

        tools = [
            ("Select/Move", "select"),
            ("Wire/Node", "wire"),
            ("Ground (0)", "ground"),
            ("Resistor", "R"),
            ("Capacitor", "C"),
            ("Inductor", "L"),
            ("Voltage Src", "V"),
            ("Current Src", "I")
        ]

        self.tool_var = tk.StringVar(value="select")
        for text, mode in tools:
            b = ttk.Radiobutton(
                tools_frame, text=text, variable=self.tool_var,
                value=mode, command=self._set_mode
            )
            b.pack(anchor="w", padx=10, pady=2)

        ttk.Separator(tools_frame).pack(fill=tk.X, pady=10)
        
        ttk.Button(tools_frame, text="Show Graph", command=self.show_graph).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tools_frame, text="Solve (Laplace)", command=self.solve_circuit).pack(fill=tk.X, padx=5, pady=2)
        # --- Simulation Time Input ---
        ttk.Label(tools_frame, text="Simulation Time (s)", font=("Arial", 9, "bold")).pack(pady=(10, 0))
        self.sim_time_var = tk.StringVar(value="1.0")
        ttk.Entry(tools_frame, textvariable=self.sim_time_var).pack(fill=tk.X, padx=5, pady=2)
        #ttk.Button(tools_frame, text="Print Indices", command=self.show_indices).pack(fill=tk.X, padx=5, pady=2)

        
        ttk.Button(tools_frame, text="Time-Domain Expressions", command=self.show_time_domain_exprs).pack(fill=tk.X, padx=5, pady=2)


        ttk.Button(tools_frame, text="Plot Time Domain", command=self.plot_results).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tools_frame, text="Clear Canvas", command=self.clear_canvas).pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(tools_frame, text="Save Schematic", command=self.save_schematic).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tools_frame, text="Load Schematic", command=self.load_schematic).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(tools_frame, text="Export .cir", command=self.export_cir_file).pack(fill=tk.X, padx=5, pady=8)
        ttk.Button(tools_frame, text="Preview .cir", command=self.preview_cir).pack(fill=tk.X, padx=5, pady=2)

        self.canvas = tk.Canvas(main_paned, bg=COLORS["bg"], cursor="crosshair")
        main_paned.add(self.canvas, weight=3)

        out_frame = ttk.Frame(self, height=150)
        out_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.output = scrolledtext.ScrolledText(out_frame, height=8, state='normal')
        self.output.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(out_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

        self._draw_grid()

    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", lambda e: self._draw_grid())

        self.bind("<Key-r>", lambda e: self._select_tool("R"))
        self.bind("<Key-c>", lambda e: self._select_tool("C"))
        self.bind("<Key-l>", lambda e: self._select_tool("L"))
        self.bind("<Key-v>", lambda e: self._select_tool("V"))
        self.bind("<Key-i>", lambda e: self._select_tool("I"))
        self.bind("<Key-w>", lambda e: self._select_tool("wire"))
        self.bind("<Key-g>", lambda e: self._select_tool("ground"))
        self.bind("<Key-s>", lambda e: self.save_schematic())

    def _select_tool(self, mode):
        self.tool_var.set(mode)
        self._set_mode()

    def _set_mode(self):
        self.mode = self.tool_var.get()
        self.temp_start_node = None
        self.status_var.set(f"Mode: {self.mode}")
        self.canvas.delete("temp")

    def _snap(self, val):
        return round(val / GRID_SIZE) * GRID_SIZE

    def _draw_grid(self):
        self.canvas.delete("grid")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        for i in range(0, w, GRID_SIZE):
            self.canvas.create_line([(i, 0), (i, h)], tag="grid", fill=COLORS["grid"])
        for i in range(0, h, GRID_SIZE):
            self.canvas.create_line([(0, i), (w, i)], tag="grid", fill=COLORS["grid"])

        self.canvas.tag_lower("grid")

    def _init_union_find(self):
        self._uf_parent = {}

    def _uf_find(self, coord):
        parent = self._uf_parent.get(coord, coord)
        if parent != coord:
            parent = self._uf_find(parent)
            self._uf_parent[coord] = parent
        return parent

    def _uf_union(self, a, b):
        ra = self._uf_find(a)
        rb = self._uf_find(b)
        if ra != rb:
            self._uf_parent[rb] = ra

    def _compress_nodes(self):
        coords = set(self.nodes.keys())
        for comp in self.components:
            coords.add(comp.n1_id)
            coords.add(comp.n2_id)

        for coord in coords:
            if coord not in self._uf_parent:
                self._uf_parent[coord] = coord

        mapping = {coord: self._uf_find(coord) for coord in coords}

        rep_name = {}
        for coord, name in list(self.nodes.items()):
            rep = mapping.get(coord, self._uf_find(coord))
            if rep in rep_name:
                if name == '0':
                    rep_name[rep] = '0'
            else:
                rep_name[rep] = name

        for rep in set(mapping.values()):
            if rep not in rep_name:
                rep_name[rep] = None

        for comp in self.components:
            comp.n1_id = mapping.get(comp.n1_id, self._uf_find(comp.n1_id))
            comp.n2_id = mapping.get(comp.n2_id, self._uf_find(comp.n2_id))

        self.nodes = {rep: name for rep, name in rep_name.items()}

    def _renumber_nodes(self):
        reps = list(self.nodes.keys())

        ground_coord = None
        for coord, name in self.nodes.items():
            if name == '0':
                ground_coord = coord
                break

        existing_nums = {}
        max_existing = 0
        for coord, name in self.nodes.items():
            if name and name != '0':
                try:
                    n = int(name)
                    existing_nums[n] = coord
                    if n > max_existing:
                        max_existing = n
                except Exception:
                    pass

        new_nodes = {}
        used_numbers = set()

        if ground_coord is not None:
            new_nodes[ground_coord] = '0'

        if getattr(self, "_ground_insert_at", None) is not None:
            shift_at = self._ground_insert_at
            remaining_coords_to_assign = []

            for coord in sorted(reps):
                if coord == ground_coord:
                    continue
                name = self.nodes.get(coord)
                if name and name != '0':
                    try:
                        n = int(name)
                        if n < shift_at:
                            new_nodes[coord] = str(n)
                            used_numbers.add(n)
                        elif n > shift_at:
                            new_nodes[coord] = str(n - 1)
                            used_numbers.add(n - 1)
                        else:
                            remaining_coords_to_assign.append(coord)
                    except Exception:
                        remaining_coords_to_assign.append(coord)
                else:
                    remaining_coords_to_assign.append(coord)

            next_id = 1
            while next_id in used_numbers:
                next_id += 1

            for coord in remaining_coords_to_assign:
                if coord in new_nodes:
                    continue
                new_nodes[coord] = str(next_id)
                used_numbers.add(next_id)
                next_id += 1
                while next_id in used_numbers:
                    next_id += 1

            self._ground_insert_at = None
            self.next_node_idx = max(used_numbers) + 1 if used_numbers else 1
            self.nodes = new_nodes
            return

        for num in sorted(existing_nums.keys()):
            coord = existing_nums[num]
            if coord == ground_coord:
                continue
            if coord in reps:
                new_nodes[coord] = str(num)
                used_numbers.add(num)

        next_id = 1
        while next_id in used_numbers:
            next_id += 1

        for coord in sorted(reps):
            if coord == ground_coord or coord in new_nodes:
                continue
            new_nodes[coord] = str(next_id)
            used_numbers.add(next_id)
            next_id += 1
            while next_id in used_numbers:
                next_id += 1

        self.nodes = new_nodes
        self.next_node_idx = next_id

    def _redraw_all(self):
        self.canvas.delete("all")
        self._draw_grid()

        for comp in self.components:
            x1, y1 = comp.n1_id
            x2, y2 = comp.n2_id
            self._draw_component_symbol(comp, x1, y1, x2, y2)

        for (nx, ny), name in self.nodes.items():
            color = "#00aa00" if name == '0' else COLORS["node"]
            r = NODE_RADIUS
            self.canvas.create_oval(nx - r, ny - r, nx + r, ny + r,
                                    fill=color, outline="black", tags=(f"node_{name}",))
            self.canvas.create_text(nx, ny - 12,
                                    text=name, font=("Arial", 8), fill="#444")

    def _draw_component_symbol(self, comp, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        same_comps = [
            c for c in self.components
            if ((c.n1_id == comp.n1_id and c.n2_id == comp.n2_id) or
                (c.n1_id == comp.n2_id and c.n2_id == comp.n1_id))
        ]

        same_comps_sorted = sorted(same_comps, key=lambda c: c.name)
        index = same_comps_sorted.index(comp)

        OFFSET_STEP = 18

        if dist != 0:
            nx = -dy / dist
            ny = dx / dist
        else:
            nx, ny = 0, 0

        mid = (len(same_comps_sorted) - 1) / 2
        offset = (index - mid) * OFFSET_STEP

        mx += nx * offset
        my += ny * offset

        def rot(x, y):
            return (x * math.cos(angle) - y * math.sin(angle) + mx,
                    x * math.sin(angle) + y * math.cos(angle) + my)

        tags = ("component", f"comp_{id(comp)}")

        if comp.type == "wire":
            self.canvas.create_line(x1, y1, x2, y2, width=2,
                                    fill=COLORS["wire"], tags=tags)
            return

        label_txt = f"{comp.name}\n{comp.value}"
        self.canvas.create_text(mx + 15, my - 15,
                                text=label_txt, font=("Arial", 9), tags=tags)

        gap = 20
        if dist < gap * 2:
            gap = dist / 2

        sx, ex = -dist / 2, dist / 2

        p2 = rot(-20, 0)
        p3 = rot(20, 0)

        self.canvas.create_line(x1, y1, p2[0], p2[1],
                                width=2, fill="black", tags=tags)
        self.canvas.create_line(p3[0], p3[1], x2, y2,
                                width=2, fill="black", tags=tags)

        if comp.type == "R":
            pts = []
            for i in range(-20, 21, 5):
                y_off = 8 if (i // 5) % 2 else -8
                rx, ry = rot(i, y_off)
                pts.extend([rx, ry])
            self.canvas.create_line(pts, width=2, fill="#a00", tags=tags)

        elif comp.type == "C":
            c1_top = rot(-4, -10)
            c1_bot = rot(-4, 10)
            c2_top = rot(4, -10)
            c2_bot = rot(4, 10)
            self.canvas.create_line(c1_top[0], c1_top[1], c1_bot[0], c1_bot[1],
                                    width=3, fill="#00a", tags=tags)
            self.canvas.create_line(c2_top[0], c2_top[1], c2_bot[0], c2_bot[1],
                                    width=3, fill="#00a", tags=tags)

        elif comp.type == "L":
            pts = []
            for i in range(-20, 20, 10):
                a = rot(i, 0)
                b = rot(i + 5, -12)
                pts.extend([a[0], a[1], b[0], b[1]])
            e = rot(20, 0)
            pts.extend([e[0], e[1]])
            self.canvas.create_line(pts, smooth=True, width=2,
                                    fill="#006400", tags=tags)

        elif comp.type in ["V", "I"]:
            r = 14
            self.canvas.create_oval(mx - r, my - r, mx + r, my + r,
                                    outline="black", width=2, tags=tags)

            if comp.type == "V":
                neg = rot(-8, 0)
                self.canvas.create_text(neg[0], neg[1], text="-",
                                        angle=math.degrees(angle),
                                        font=("Arial", 12, "bold"),
                                        tags=tags)
                pos = rot(8, 0)
                self.canvas.create_text(pos[0], pos[1], text="+",
                                        angle=math.degrees(angle),
                                        font=("Arial", 12, "bold"),
                                        tags=tags)
            elif comp.type == "I":
                a1 = rot(-8, 0)
                a2 = rot(8, 0)
                self.canvas.create_line(a1[0], a1[1], a2[0], a2[1],
                                        arrow=tk.LAST, width=2, tags=tags)

    def _get_node_at(self, x, y):
        return self.nodes.get((x, y))

    def _create_node_if_missing(self, x, y):
        coord = (x, y)
        if coord not in self.nodes:
            name = str(self.next_node_idx)
            self.nodes[coord] = name
            self.next_node_idx += 1
            if coord not in self._uf_parent:
                self._uf_parent[coord] = coord
            return name
        return self.nodes[coord]

    def _on_click(self, event):
        x, y = self._snap(event.x), self._snap(event.y)

        if self.mode == "select":
            return

        if self.mode == "ground":
            if (x, y) in self.nodes and self.nodes[(x, y)] != '0':
                old_name = self.nodes[(x, y)]
                try:
                    self._ground_insert_at = int(old_name)
                except Exception:
                    self._ground_insert_at = None
                self.nodes[(x, y)] = '0'
                if (x, y) not in self._uf_parent:
                    self._uf_parent[(x, y)] = (x, y)
                self._compress_nodes()
                self._renumber_nodes()
                self._redraw_all()
                self.log("Ground set here.")
                return

            created_name = None
            if (x, y) not in self.nodes:
                created_name = self._create_node_if_missing(x, y)
            try:
                if created_name is None:
                    existing = self.nodes.get((x, y))
                    if existing and existing != '0':
                        self._ground_insert_at = int(existing)
                    else:
                        self._ground_insert_at = None
                else:
                    self._ground_insert_at = int(created_name)
            except Exception:
                self._ground_insert_at = None

            self.nodes[(x, y)] = '0'
            self._compress_nodes()
            self._renumber_nodes()
            self._redraw_all()
            self.log("Ground set here.")
            return

        if not self.temp_start_node:
            self.temp_start_node = (x, y)
            self._create_node_if_missing(x, y)
            r = 3
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", tags="temp")
        else:
            start_pos = self.temp_start_node
            end_pos = (x, y)

            if start_pos == end_pos:
                self.temp_start_node = None
                self.canvas.delete("temp")
                return

            self._finish_placement(start_pos, end_pos)

    def _on_drag(self, event):
        if self.temp_start_node:
            x, y = self._snap(event.x), self._snap(event.y)
            self.canvas.delete("drag_line")
            sx, sy = self.temp_start_node
            self.canvas.create_line(sx, sy, x, y, dash=(4, 2),
                                    tags="drag_line", fill="gray")

    def _on_release(self, event):
        pass

    def _finish_placement(self, p1, p2):
        self.canvas.delete("temp")
        self.canvas.delete("drag_line")
        self.temp_start_node = None

        self._create_node_if_missing(p1[0], p1[1])
        self._create_node_if_missing(p2[0], p2[1])

        if self.mode == "wire":
            self._uf_union(p1, p2)
            comp = ComponentItem("wire", f"w{len(self.components) + 1}", 0, p1, p2)
            self.components.append(comp)
            self._compress_nodes()
            self._renumber_nodes()
            self._redraw_all()

        elif self.mode in self.comp_counters:
            val_str = simpledialog.askstring("Value", f"Enter value for {self.mode}:",
                                             initialvalue="1000" if self.mode == "R" else "1")
            if val_str:
                try:
                    val = float(val_str)
                except Exception:
                    val = val_str

                name = f"{self.mode}{self.comp_counters[self.mode]}"
                self.comp_counters[self.mode] += 1

                comp = ComponentItem(self.mode, name, val, p1, p2)
                self.components.append(comp)

            self._compress_nodes()
            self._renumber_nodes()
            self._redraw_all()

    def _on_right_click(self, event):
        items = self.canvas.find_closest(event.x, event.y)
        if not items:
            return

        item_id = items[0]
        tags = self.canvas.gettags(item_id)

        clicked_comp = None
        for tag in tags:
            if tag.startswith("comp_"):
                try:
                    mem_id = int(tag.split("_", 1)[1])
                except Exception:
                    continue
                for c in self.components:
                    if id(c) == mem_id:
                        clicked_comp = c
                        break
                if clicked_comp:
                    break

        if clicked_comp:
            m = tk.Menu(self, tearoff=0)
            m.add_command(label=f"Edit {clicked_comp.name}",
                          command=lambda: self._edit_comp(clicked_comp))
            m.add_command(label="Delete", command=lambda: self._delete_comp(clicked_comp))
            try:
                m.tk_popup(event.x_root, event.y_root)
            finally:
                m.grab_release()

    def _delete_comp(self, comp):
        if comp in self.components:
            self.components.remove(comp)
            self._compress_nodes()
            self._renumber_nodes()
            self._redraw_all()

    def _edit_comp(self, comp):
        new_val = simpledialog.askstring("Edit", f"New value for {comp.name}:",
                                         initialvalue=str(comp.value))
        if new_val:
            try:
                comp.value = float(new_val)
            except Exception:
                comp.value = new_val
            self._compress_nodes()
            self._renumber_nodes()
            self._redraw_all()

    def clear_canvas(self):
        self.components = []
        self.nodes = {}
        self.next_node_idx = 1
        self.comp_counters = {"R": 1, "C": 1, "L": 1, "V": 1, "I": 1}
        self._init_union_find()
        self._ground_insert_at = None
        self._redraw_all()

    def save_schematic(self):
        fn = filedialog.asksaveasfilename(defaultextension=".json",
                                          filetypes=[("JSON files", "*.json")])
        if not fn:
            return
        data = {
            "nodes": [{"coord": coord, "name": name}
                      for coord, name in self.nodes.items()],
            "components": [
                {"type": c.type, "name": c.name, "value": c.value,
                 "n1": c.n1_id, "n2": c.n2_id}
                for c in self.components
            ]
        }
        try:
            with open(fn, "w") as f:
                json.dump(data, f, indent=2)
            self.log(f"Saved schematic to {fn}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def load_schematic(self):
        fn = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not fn:
            return
        try:
            with open(fn, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.nodes = {tuple(n["coord"]): n["name"]
                      for n in data.get("nodes", [])}
        self.components = []
        for c in data.get("components", []):
            comp = ComponentItem(c["type"], c["name"], c["value"],
                                 tuple(c["n1"]), tuple(c["n2"]))
            self.components.append(comp)

        self._init_union_find()

        for coord in list(self.nodes.keys()):
            self._uf_parent[coord] = coord
        for comp in self.components:
            if comp.n1_id not in self._uf_parent:
                self._uf_parent[comp.n1_id] = comp.n1_id
            if comp.n2_id not in self._uf_parent:
                self._uf_parent[comp.n2_id] = comp.n2_id

        self._compress_nodes()
        self._ground_insert_at = None
        self._renumber_nodes()
        self._redraw_all()
        self.log(f"Loaded schematic from {fn}")

    def _node_name(self, coord):
        if coord in self.nodes and self.nodes[coord] is not None:
            return self.nodes[coord]

        rep = self._uf_find(coord)
        if rep in self.nodes and self.nodes[rep] is not None:
            return self.nodes[rep]

        tmp = str(self.next_node_idx)
        self.next_node_idx += 1
        self.nodes[coord] = tmp
        return tmp

    def format_value_for_spice(self, val):
        try:
            return str(float(val))
        except Exception:
            return str(val)

    def get_cir_text(self, title="Generated by SchematicEditor"):
        self._compress_nodes()
        self._renumber_nodes()

        lines = []

        for c in self.components:
            if c.type == "wire":
                continue
            n1 = self._node_name(c.n1_id)
            n2 = self._node_name(c.n2_id)
            val_str = self.format_value_for_spice(c.value)

            if c.type in ("R", "C", "L"):
                lines.append(f"{c.type} {c.name} {n1} {n2} {val_str} ;")
            elif c.type == "V":
                lines.append(f"V {c.name} {n2} {n1}  {val_str} ;")
            elif c.type == "I":
                lines.append(f"I {c.name} {n1} {n2}  {val_str} ;")
            else:
                lines.append(f"* Unknown component type {c.type} for {c.name} ;")
                lines.append(f"{c.type} {c.name} {n1} {n2} {val_str};")

        return "\n".join(lines)

    def export_cir_file(self):
        cir_text = self.get_cir_text()
        fn = filedialog.asksaveasfilename(defaultextension=".cir",
                                          filetypes=[("SPICE files", "*.cir"),
                                                     ("Text files", "*.txt")])
        if not fn:
            self.log("Export cancelled.")
            return cir_text
        try:
            with open(fn, "w") as f:
                f.write(cir_text)
            self.log(f"Exported .cir file to {fn}")
            return cir_text
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            return cir_text

    def preview_cir(self):
        cir_text = self.get_cir_text()

        top = tk.Toplevel(self)
        top.title(".cir Preview")

        txt = scrolledtext.ScrolledText(top, width=100, height=30)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", cir_text)
        txt.configure(state="normal")

        ttk.Button(
            top, text="Save As...",
            command=lambda: (self._save_from_preview(txt.get("1.0", tk.END)), top.lift())
        ).pack(pady=4)

    def _save_from_preview(self, text):
        fn = filedialog.asksaveasfilename(defaultextension=".cir",
                                          filetypes=[("SPICE files", "*.cir"),
                                                     ("Text files", "*.txt")])
        if not fn:
            return
        try:
            with open(fn, "w") as f:
                f.write(text)
            self.log(f"Saved preview .cir to {fn}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _safe_simplify(self, expr):
        try:
            return sp.simplify(expr)
        except Exception:
            return expr
    def _get_parsed_circuit(self):
        cir = self.get_cir_text()
        path = self._build_temp_netlist(cir)
        return parse_netlist(path)
    def show_matrices(self):
        if not BACKEND_AVAILABLE:
            self.log("Backend not available.")
            return

        circ = self._get_parsed_circuit()
        s = sp.symbols('s')
        Ared, names_all, ref_idx = circ.reduced_incidence()
        Bmat, tree_edges, chords = circ.fundamental_tieset()

        top = tk.Toplevel(self)
        top.title("Circuit Matrices")
        txt = scrolledtext.ScrolledText(top, width=120, height=35)
        txt.pack(fill=tk.BOTH, expand=True)

        def write(name, M):
            txt.insert(tk.END, f"\n=== {name} ===\n")
            arr = np.array(M.tolist(), dtype=object)
            for row in arr:
                txt.insert(tk.END, " ".join([str(x) for x in row]) + "\n")
        write("Reduced incidence matrix A_red", Ared)
        write("Tie-set matrix B", Bmat)

        txt.insert(tk.END, "\nTree edges:\n")
        for e in tree_edges:
            txt.insert(tk.END, f"  {e}\n")
        txt.insert(tk.END, "\nChords:\n")
        for c in chords:
            txt.insert(tk.END, f"  {c}\n")

        txt.configure(state="disabled")


    def show_graph(self):
        if not BACKEND_AVAILABLE:
            self.log("Backend not available.")
            return

        circ = self._get_parsed_circuit()
        try:
            circ.draw_graph()
            self.log("Graph drawn in separate window.")
        except Exception as e:
            self.log(f"Graph error: {e}")
    def _detect_floating_vsource(self, circ):
        ref = getattr(circ, 'ref_node', None)
        for br in getattr(circ, 'branches', []):
            btype = getattr(br, 'btype', None)
            n1 = getattr(br, 'n1', None)
            n2 = getattr(br, 'n2', None)
            if btype == 'V' and ref is not None and n1 != ref and n2 != ref:
                return True
        return False
    def show_time_domain_exprs(self):
        if not hasattr(self, 'last_result'):
            self.log("Solve first.")
            return

        s, t = sp.symbols('s t')
        res = self.last_result
        top = tk.Toplevel(self)
        top.title("Time-Domain Expressions")
        txt = scrolledtext.ScrolledText(top, width=100, height=35)
        txt.pack(fill=tk.BOTH, expand=True)

    # Node voltages
        txt.insert(tk.END, "=== Node Voltages (time domain) ===\n")
        if res.get("Vn"):
            for i, expr in enumerate(res["Vn"]):
                try:
                    f_t = sp.inverse_laplace_transform(expr, s, t)
                except:
                    f_t = "Cannot invert"
                txt.insert(tk.END, f"n{i}(t) = {sp.simplify(f_t)}\n\n")
        else:
            txt.insert(tk.END, "(No node voltages)\n")

    # Currents
        txt.insert(tk.END, "\n=== Branch Currents (time domain) ===\n")
        if res.get("Ib"):
            for i, expr in enumerate(res["Ib"]):
                try:
                    f_t = sp.inverse_laplace_transform(expr, s, t)
                except:
                    f_t = "Cannot invert"
                txt.insert(tk.END, f"Ib{i}(t) = {sp.simplify(f_t)}\n\n")
        else:
            txt.insert(tk.END, "(No branch currents)\n")

        txt.configure(state="normal")

    def _normalize_solver_result(self, res):
        out = {}
        if 'Vn' in res:
            out['Vn'] = list(res['Vn'])
        else:
            out['Vn'] = []

        if 'Ib' in res:
            out['Ib'] = list(res['Ib'])
        elif 'I_branch' in res:
            out['Ib'] = list(res['I_branch'])
        elif 'I_b' in res:
            out['Ib'] = list(res['I_b'])
        else:
            out['Ib'] = []

        out['names'] = res.get('names', [])
        return out

    def _build_temp_netlist(self, cir_text):
        fd, path = tempfile.mkstemp(suffix='.cir', text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(cir_text)
        return path
    def _solve_backend_from_cir(self, cir_text):
        if not BACKEND_AVAILABLE:
            self.log("Backend not available.")
            return None

        fd = None
        path = None
        try:
            fd, path = tempfile.mkstemp(suffix=".cir", text=True)
            with os.fdopen(fd, "w") as f:
                f.write(cir_text)
                fd = None

            try:
                circ = parse_netlist(path)
            except Exception as e:
                self.log(f"Netlist parse error: {e}")
                return None

            s, t = sp.symbols('s t')

            floating_vs = False
            try:
                ref = getattr(circ, 'ref_node', None)
                for br in getattr(circ, 'branches', []):
                    if getattr(br, 'btype', None) == 'V':
                        n1 = getattr(br, 'n1', None)
                        n2 = getattr(br, 'n2', None)
                        if ref is not None and n1 != ref and n2 != ref:
                            floating_vs = True
                            break
            except Exception:
                floating_vs = True

            try:
                if not floating_vs:
                    raw = solve_node_symbolic(circ, s)
                    # --- Add branch metadata for labeling plots ---
                    raw["branch_labels"] = [br.name for br in circ.branches]
                    raw["branch_types"]  = [br.btype for br in circ.branches]
                    raw["branch_nodes"]  = [(br.n1, br.n2) for br in circ.branches]

                else:
                    raw = solve_mna_symbolic(circ, s)
                    # --- Add branch metadata for labeling plots ---
                    raw["branch_labels"] = [br.name for br in circ.branches]
                    raw["branch_types"]  = [br.btype for br in circ.branches]
                    raw["branch_nodes"]  = [(br.n1, br.n2) for br in circ.branches]

            except Exception as e:
                self.log(f"Symbolic solver error: {e}")
                return None

            try:
                return self._normalize_solver_result(raw)
            except Exception as e:
                self.log(f"Result normalization error: {e}")
                return None

        finally:
            try:
                if fd is not None:
                    os.close(fd)
            except Exception:
                pass
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def solve_circuit(self):
        self.log("Building SPICE netlist...")
        cir_text = self.get_cir_text()

        self.log("Running symbolic analysis...")
        res = self._solve_backend_from_cir(cir_text)

        if res is None:
            self.log("Solver returned no result.")
            return

        self.last_result = res

        self.log("--- SOLUTION (Laplace Domain) ---")

        if res.get('Vn'):
            for i, expr in enumerate(res['Vn']):
                name = res.get('names', [])[i] if i < len(res.get('names', [])) else f"n{i}"
                try:
                    pretty = self._safe_simplify(expr)
                except Exception:
                    pretty = expr
                self.log(f"V({name}) = {pretty}")
        else:
            self.log("(Node voltages not available)")

        if res.get('Ib'):
            for i, expr in enumerate(res['Ib']):
                try:
                    pretty = self._safe_simplify(expr)
                except Exception:
                    pretty = expr
                self.log(f"Ib{i} = {pretty}")
        else:
            self.log("(Branch currents not available)")

    def plot_results(self):
        if not BACKEND_AVAILABLE:
            self.log("Backend not available.")
            return

        if not hasattr(self, 'last_result'):
            self.log("Nothing to plot. Solve first.")
            return
        try:
            t_end = float(self.sim_time_var.get())
        except Exception:
            self.log("Invalid simulation time entered.")
            return
        res = self.last_result
        s, t = sp.symbols('s t')

        if res.get('Vn'):
            try:
                inverse_and_plot(res['Vn'], s, t, title_prefix="n", ylabel="Voltage (V)",t_end=t_end)
                self.log("Plotted node voltages.")
            except Exception as e:
                self.log(f"Plot Error (Voltages): {e}")
        else:
            self.log("No node voltages available to plot.")

        if res.get('Ib'):
            try:
                labels = []
                if res.get("branch_labels"):
                    for i, name in enumerate(res["branch_labels"]):
                        btype = res["branch_types"][i]
                        n1, n2 = res["branch_nodes"][i]
                        labels.append(f"{name} ({btype}: {n1}->{n2})")
                else:
                    labels = None
                inverse_and_plot(res['Ib'], s, t, title_prefix="Ib", ylabel="Current (A)", series_labels=labels, t_end=t_end)
                self.log("Plotted branch currents.")
            except Exception as e:
                self.log(f"Plot Error (Currents): {e}")
        else:
            self.log("No branch currents available to plot.")

    def log(self, msg):
        self.output.insert(tk.END, str(msg) + "\n")
        self.output.see(tk.END)
