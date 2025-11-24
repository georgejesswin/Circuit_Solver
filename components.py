# components.py

GRID_SIZE = 20
NODE_RADIUS = 5

COLORS = {
    "bg": "#f0f0f0",
    "grid": "#d0d0d0",
    "wire": "#000000",
    "node": "#000080",
    "select": "#ff00ff",
    "text": "#000000"
}

class ComponentItem:
    def __init__(self, c_type, name, value, n1_id, n2_id):
        self.type = c_type
        self.name = name
        self.value = value
        self.n1_id = n1_id
        self.n2_id = n2_id
