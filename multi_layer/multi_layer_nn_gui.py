import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: int

class MultiLayerNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Layer Neural Network")
        self.root.geometry("800x600")
        
        self.points = []
        self.num_classes = 2
        self.current_class = 1
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Main Layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for Coordinate System
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Controls Frame
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, anchor="n")
        
        # Class Count Selection
        tk.Label(controls_frame, text="Class Count:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_count_var = tk.IntVar(value=2)
        self.class_count_combo = ttk.Combobox(controls_frame, textvariable=self.class_count_var, values=[2, 3, 4, 5, 6], state="readonly", width=5)
        self.class_count_combo.pack(anchor="w")
        self.class_count_combo.bind("<<ComboboxSelected>>", self.on_class_count_change)
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        # Class Selection
        tk.Label(controls_frame, text="Select Class:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_select_var = tk.StringVar(value="Class 1")
        self.class_select_combo = ttk.Combobox(controls_frame, textvariable=self.class_select_var, state="readonly", width=10)
        self.class_select_combo.pack(anchor="w")
        self.class_select_combo.bind("<<ComboboxSelected>>", self.on_class_select_change)
        self.update_class_select_options()
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        tk.Label(controls_frame, text="Number of Hidden Layers:").pack(anchor="w", pady=(0, 5))
        self.num_layers_var = tk.IntVar(value=1)
        tk.Entry(controls_frame, textvariable=self.num_layers_var).pack(anchor="w", pady=(0, 10))

        tk.Label(controls_frame, text="Neurons per Layer (e.g. 4,5):").pack(anchor="w", pady=(0, 5))
        self.hidden_layers_var = tk.StringVar(value="5")
        tk.Entry(controls_frame, textvariable=self.hidden_layers_var).pack(anchor="w", pady=(0, 10))

        # Normalize Checkbox
        self.normalize_var = tk.BooleanVar()
        tk.Checkbutton(controls_frame, text="Normalize Data", variable=self.normalize_var).pack(anchor="w", pady=(0, 10))

        tk.Label(controls_frame, text="Max Epochs:").pack(anchor="w", pady=(0, 5))
        self.max_epochs_var = tk.IntVar(value=1000)
        tk.Entry(controls_frame, textvariable=self.max_epochs_var).pack(anchor="w", pady=(0, 10))

        tk.Label(controls_frame, text="Learning Rate:").pack(anchor="w", pady=(0, 5))
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        tk.Entry(controls_frame, textvariable=self.learning_rate_var).pack(anchor="w", pady=(0, 10))

        tk.Label(controls_frame, text="Min Error:").pack(anchor="w", pady=(0, 5))
        self.min_error_var = tk.DoubleVar(value=0.01)
        tk.Entry(controls_frame, textvariable=self.min_error_var).pack(anchor="w", pady=(0, 10))

        tk.Button(controls_frame, text="Show Error Graph", command=self.show_error_graph).pack(anchor="w", pady=5)
        tk.Button(controls_frame, text="Show Regression", command=self.show_regression_graph).pack(anchor="w", pady=5)

        # Draw Axes
        self.draw_axes()
        
        # Setup Menu
        self.setup_menu()

    def on_class_count_change(self, event):
        self.num_classes = self.class_count_var.get()
        self.update_class_select_options()
        self.points = [] 
        self.draw_canvas()

    def update_class_select_options(self):
        options = [f"Class {i}" for i in range(1, self.num_classes + 1)]
        self.class_select_combo['values'] = options
        if self.current_class > self.num_classes:
            self.current_class = 1
            self.class_select_var.set("Class 1")

    def on_class_select_change(self, event):
        selected = self.class_select_var.get()
        self.current_class = int(selected.split(" ")[1])

    def screen_to_cartesian(self, sx, sy):
        scale = 20 / self.canvas_width
        cx = (sx - self.canvas_width/2) * scale
        cy = (self.canvas_height/2 - sy) * scale
        return cx, cy

    def cartesian_to_screen(self, cx, cy):
        scale = self.canvas_width / 20
        sx = cx * scale + self.canvas_width/2
        sy = self.canvas_height/2 - cy * scale
        return sx, sy

    def on_canvas_click(self, event):
        cx, cy = self.screen_to_cartesian(event.x, event.y)
        self.points.append(Point(cx, cy, self.current_class))
        self.draw_point(event.x, event.y, self.current_class)
        
    def draw_point(self, x, y, label):
        r = 4
        color = self.colors[(label - 1) % len(self.colors)]
        self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, fill=color)

    def draw_canvas(self):
        self.canvas.delete("all")
        self.draw_axes()
        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            self.draw_point(sx, sy, p.label)
                
    def setup_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        
        process_menu.add_command(label="Randomly Weights", command=self.initialize_weights)
        
        train_menu = tk.Menu(process_menu, tearoff=0)
        process_menu.add_cascade(label="Train", menu=train_menu)
        
        train_menu.add_command(label="With Momentum", command=self.train_with_momentum)
        train_menu.add_command(label="Without Momentum", command=self.train_without_momentum)
        
        process_menu.add_command(label="Test", command=self.test_network)

    def initialize_weights(self):
        print("Initialize Weights Randomly clicked")

    def train_with_momentum(self):
        print("Train With Momentum clicked")

    def train_without_momentum(self):
        print("Train Without Momentum clicked")

    def test_network(self):
        print("Test Network clicked")

    def show_error_graph(self):
        print("Show Error Graph clicked")

    def show_regression_graph(self):
        print("Show Regression Graph clicked")
        
    def draw_axes(self):
        cw = self.canvas_width
        ch = self.canvas_height
        
        # X1 Axis (Horizontal)
        self.canvas.create_line(0, ch/2, cw, ch/2, fill="black", width=2)
        self.canvas.create_text(cw - 20, ch/2 + 20, text="x1", font=("Arial", 12, "bold"))
        
        # X2 Axis (Vertical)
        self.canvas.create_line(cw/2, 0, cw/2, ch, fill="black", width=2)
        self.canvas.create_text(cw/2 + 20, 20, text="x2", font=("Arial", 12, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLayerNNGUI(root)
    root.mainloop()
