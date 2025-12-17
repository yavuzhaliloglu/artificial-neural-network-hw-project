import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import random
import time
import math
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: int

class MultiClassNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Class Neural Network - Single Layer")
        self.root.geometry("900x600")
        
        # Data
        self.points = [] 
        self.num_classes = 2
        self.current_class = 1
        self.weights_matrix = [] # Matrix K x N (Rows x Cols)
        self.learning_rate = 0.5
        self.bias = 1.0
        self.errors = []
        
        self.is_normalized = False
        self.norm_params = {}
        
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # UI Layout
        self.setup_ui()
        self.initialize_weights()
        
    def setup_ui(self):
        # Menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processes", menu=process_menu)
        
        # Initializing Submenu
        init_menu = tk.Menu(process_menu, tearoff=0)
        process_menu.add_cascade(label="Initializing", menu=init_menu)
        init_menu.add_command(label="Randomly", command=self.initialize_randomly)
        
        process_menu.add_separator()
        
        # Training Submenu
        train_menu = tk.Menu(process_menu, tearoff=0)
        process_menu.add_cascade(label="Training", menu=train_menu)
        train_menu.add_command(label="Discrete (Perceptron)", command=self.train_discrete)
        train_menu.add_command(label="Continuous (Delta)", command=self.train_continuous)
        
        process_menu.add_separator()
        process_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main Layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas (Left)
        self.canvas_width = 600
        self.canvas_height = 550
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Controls (Right)
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10, anchor="n")
        
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

        # Normalize Checkbox
        self.normalize_var = tk.BooleanVar()
        tk.Checkbutton(controls_frame, text="Normalize Data", variable=self.normalize_var).pack(anchor="w")

        tk.Label(controls_frame, text="").pack() # Spacer

        # Learning Rate
        tk.Label(controls_frame, text="Learning Rate:", font=("Arial", 10)).pack(anchor="w")
        self.learning_rate_var = tk.DoubleVar(value=0.5)
        tk.Entry(controls_frame, textvariable=self.learning_rate_var, width=10).pack(anchor="w")

        # Max Epochs
        tk.Label(controls_frame, text="Max Epochs:", font=("Arial", 10)).pack(anchor="w")
        self.max_epochs_var = tk.IntVar(value=1000)
        tk.Entry(controls_frame, textvariable=self.max_epochs_var, width=10).pack(anchor="w")

        # Min Error
        tk.Label(controls_frame, text="Min Error:", font=("Arial", 10)).pack(anchor="w")
        self.min_error_var = tk.DoubleVar(value=0.01)
        tk.Entry(controls_frame, textvariable=self.min_error_var, width=10).pack(anchor="w")
        
        tk.Button(controls_frame, text="Show Error Graph", command=self.show_error_graph).pack(anchor="w", pady=10)
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        # Info Display
        self.cycle_label = tk.Label(controls_frame, text="Cycles: 0", font=("Arial", 10))
        self.cycle_label.pack(anchor="w", pady=(10, 0))
        
        self.draw_axes()

    def initialize_weights(self):
        # Matrix K x N (Rows x Cols)
        # K = num_classes
        # N = 3 (Bias, x, y)
        self.weights_matrix = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(self.num_classes)]

    def on_class_count_change(self, event):
        self.num_classes = self.class_count_var.get()
        self.update_class_select_options()
        self.initialize_weights()
        self.points = [] # Clear points on class count change to avoid confusion
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

    def draw_axes(self):
        cw = self.canvas_width
        ch = self.canvas_height
        self.canvas.create_line(cw/2, 0, cw/2, ch, fill="black", width=2) # x2 axis
        self.canvas.create_line(0, ch/2, cw, ch/2, fill="black", width=2) # x1 axis
        
        self.canvas.create_text(cw - 20, ch/2 + 20, text="x1", font=("Arial", 12, "bold"))
        self.canvas.create_text(cw/2 + 20, 20, text="x2", font=("Arial", 12, "bold"))
        
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
        self.draw_decision_boundaries()

    def initialize_randomly(self):
        self.initialize_weights()
        self.draw_canvas()

    def draw_decision_boundaries(self):
        self.canvas.delete("boundary")
        
        for i in range(self.num_classes):
            w = self.weights_matrix[i]
            cls = i + 1
            w0, w1, w2 = w
            color = self.colors[(cls - 1) % len(self.colors)]
            
            if w2 == 0:
                if w1 == 0: continue
                
                if self.is_normalized and self.norm_params:
                    x_train = -w0 * self.bias / w1
                    x_val = (x_train + 1) * self.norm_params['x_range'] / 2 + self.norm_params['x_min']
                else:
                    x_val = -w0 * self.bias / w1
                    
                sx1, sy1 = self.cartesian_to_screen(x_val, 10)
                sx2, sy2 = self.cartesian_to_screen(x_val, -10)
            else:
                x1_bound, x2_bound = -10, 10
                
                if self.is_normalized and self.norm_params:
                    x_min = self.norm_params['x_min']
                    x_range = self.norm_params['x_range']
                    y_min = self.norm_params['y_min']
                    y_range = self.norm_params['y_range']
                    
                    def get_y_screen(x_screen):
                        x_train = 2 * (x_screen - x_min) / x_range - 1
                        y_train = (-w1 * x_train - w0 * self.bias) / w2
                        y_screen = (y_train + 1) * y_range / 2 + y_min
                        return y_screen

                    y1 = get_y_screen(x1_bound)
                    y2 = get_y_screen(x2_bound)
                else:
                    y1 = (-w1 * x1_bound - w0 * self.bias) / w2
                    y2 = (-w1 * x2_bound - w0 * self.bias) / w2
                
                sx1, sy1 = self.cartesian_to_screen(x1_bound, y1)
                sx2, sy2 = self.cartesian_to_screen(x2_bound, y2)
                
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill=color, width=2, tags="boundary")

    def prepare_data(self):
        if not self.points:
            return None
            
        data = []
        for p in self.points:
            data.append({'x': p.x, 'y': p.y, 'label': p.label})
            
        if self.is_normalized:
            xs = [d['x'] for d in data]
            ys = [d['y'] for d in data]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0
            
            self.norm_params = {
                'x_min': x_min, 'x_range': x_range,
                'y_min': y_min, 'y_range': y_range
            }
            
            for d in data:
                d['x'] = 2 * (d['x'] - x_min) / x_range - 1
                d['y'] = 2 * (d['y'] - y_min) / y_range - 1
        else:
            self.norm_params = {}
            
        return data

    def train_discrete(self):
        print("Discrete training (Perceptron - One vs All) selected")
        
        try:
            max_epochs = self.max_epochs_var.get()
            learning_rate = self.learning_rate_var.get()
            min_error = self.min_error_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check Learning Rate, Max Epochs, and Min Error.")
            return

        self.cycle_count = 0
        self.errors = []
        
        self.is_normalized = self.normalize_var.get()
        training_data = self.prepare_data()
        
        if not training_data:
            messagebox.showwarning("Warning", "No data points to train!")
            return

        for epoch in range(max_epochs):
            global_error = 0
            
            for data in training_data:
                # Input Vector y (using user's notation y for input)
                # y = [bias, x, y] (or [1, x1, x2])
                inputs = [self.bias, data['x'], data['y']]
                
                # Desired Output Vector d
                # If class is 1-based index in data['label']
                d = [-1] * self.num_classes
                d[data['label'] - 1] = 1 
                
                # Net Vector Calculation: net = W * y
                net = []
                for k in range(self.num_classes):
                    val = sum(self.weights_matrix[k][j] * inputs[j] for j in range(3))
                    net.append(val)
                    
                # Output Vector o: o = sgn(net)
                o = [1 if n >= 0 else -1 for n in net]
                
                # Error Vector e: e = d - o
                e = [d[k] - o[k] for k in range(self.num_classes)]
                
                # Update Weights Matrix: W = W + c * e * y^T
                # W_kj = W_kj + c * e_k * y_j
                for k in range(self.num_classes):
                    if e[k] != 0:
                        global_error += 1 # Count errors (scalar for stopping condition)
                        for j in range(3):
                            self.weights_matrix[k][j] += learning_rate * e[k] * inputs[j]
            
            self.errors.append(global_error)
            self.cycle_count += 1
            self.draw_canvas()
            self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
            self.root.update()
            # time.sleep(0.005) # Optional delay
            
            if global_error <= min_error:
                print(f"Converged in {epoch+1} epochs.")
                break
        else:
             print("Did not converge within max epochs.")

    def train_continuous(self):
        print("Continuous training (Delta - One vs All) selected")
        
        try:
            max_epochs = self.max_epochs_var.get()
            learning_rate = self.learning_rate_var.get()
            min_error = self.min_error_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check Learning Rate, Max Epochs, and Min Error.")
            return

        self.cycle_count = 0
        self.errors = []
        
        self.is_normalized = self.normalize_var.get()
        training_data = self.prepare_data()
        
        if not training_data:
            messagebox.showwarning("Warning", "No data points to train!")
            return

        for epoch in range(max_epochs):
            total_error = 0
            
            for data in training_data:
                # Input Vector y
                inputs = [self.bias, data['x'], data['y']]
                
                # Desired Output Vector d
                d = [0] * self.num_classes
                d[data['label'] - 1] = 1
                
                # Net Vector Calculation: net = W * y
                net = []
                for k in range(self.num_classes):
                    val = sum(self.weights_matrix[k][j] * inputs[j] for j in range(3))
                    net.append(val)
                    
                # Output Vector o: o = f(net) (Sigmoid)
                o = []
                for n in net:
                    try:
                        res = 1 / (1 + math.exp(-n))
                    except OverflowError:
                        res = 0 if n < 0 else 1
                    o.append(res)
                
                # Error and Weight Update
                for k in range(self.num_classes):
                    error_k = d[k] - o[k]
                    total_error += error_k ** 2
                    
                    derivative = o[k] * (1 - o[k])
                    delta = error_k * derivative
                    
                    for j in range(3):
                        self.weights_matrix[k][j] += learning_rate * delta * inputs[j]
            
            self.errors.append(total_error)
            self.cycle_count += 1
            self.draw_canvas()
            self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
            self.root.update()
            # time.sleep(0.005)
            
            if total_error < min_error:
                print(f"Converged in {epoch+1} epochs. Total Error: {total_error}")
                break

    def show_error_graph(self):
        if not self.errors:
            messagebox.showinfo("Info", "No error data to display.")
            return
            
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Error Graph")
        graph_window.geometry("600x400")
        
        canvas = tk.Canvas(graph_window, bg="white", width=550, height=350)
        canvas.pack(padx=20, pady=20)
        
        # Draw axes
        canvas.create_line(50, 300, 500, 300, width=2) # X axis
        canvas.create_line(50, 300, 50, 50, width=2)   # Y axis
        
        # Labels
        canvas.create_text(275, 330, text="Epochs")
        canvas.create_text(20, 175, text="Error", angle=90)
        
        max_epoch = len(self.errors)
        max_error = max(self.errors) if self.errors else 1
        if max_error == 0: max_error = 1
        
        # Draw ticks and numbers
        # Y axis (Error)
        for i in range(6):
            y_val = max_error * i / 5
            y_pos = 300 - (i / 5) * 250
            canvas.create_line(45, y_pos, 50, y_pos)
            canvas.create_text(40, y_pos, text=f"{y_val:.2f}", anchor="e", font=("Arial", 8))

        # X axis (Epochs)
        for i in range(6):
            x_val = max_epoch * i / 5
            x_pos = 50 + (i / 5) * 450
            canvas.create_line(x_pos, 300, x_pos, 305)
            canvas.create_text(x_pos, 315, text=f"{int(x_val)}", anchor="n", font=("Arial", 8))

        # Plot
        points = []
        for i, error in enumerate(self.errors):
            x = 50 + (i / max_epoch) * 450 if max_epoch > 0 else 50
            y = 300 - (error / max_error) * 250
            points.append(x)
            points.append(y)
            
        if len(points) >= 4:
            canvas.create_line(*points, fill="blue", width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiClassNNGUI(root)
    root.mainloop()
