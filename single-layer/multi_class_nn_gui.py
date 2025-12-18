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
        self.root.title("Single-Layer Neural Network - Classification & Regression")
        self.root.geometry("1000x800")
        
        # Data
        self.points = [] 
        self.num_classes = 2
        self.current_class = 1
        self.mode = "Manual"  # "Manual" (Classification) or "Regression"
        self.weights_matrix = [] # Matrix K x N (Rows x Cols)
        self.learning_rate = 0.5
        self.bias = 1.0
        self.errors = []
        self.training_results = []
        
        self.is_normalized = False
        self.norm_params = {}
        self.normalization_params_in = None
        self.normalization_params_out = None
        
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
        
        # Mode Selection
        tk.Label(controls_frame, text="Mode:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.mode_var = tk.StringVar(value="Manual")
        tk.Radiobutton(controls_frame, text="Classification", variable=self.mode_var, value="Manual", command=self.toggle_mode).pack(anchor="w")
        tk.Radiobutton(controls_frame, text="Regression (Curve Fitting)", variable=self.mode_var, value="Regression", command=self.toggle_mode).pack(anchor="w")
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        # Classification Controls Frame
        self.class_controls_frame = tk.Frame(controls_frame)
        
        # Class Count Selection
        tk.Label(self.class_controls_frame, text="Class Count:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_count_var = tk.IntVar(value=2)
        self.class_count_combo = ttk.Combobox(self.class_controls_frame, textvariable=self.class_count_var, values=[2, 3, 4, 5, 6], state="readonly", width=5)
        self.class_count_combo.pack(anchor="w")
        self.class_count_combo.bind("<<ComboboxSelected>>", self.on_class_count_change)
        
        tk.Label(self.class_controls_frame, text="").pack() # Spacer
        
        # Class Selection
        tk.Label(self.class_controls_frame, text="Select Class:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_select_var = tk.StringVar(value="Class 1")
        self.class_select_combo = ttk.Combobox(self.class_controls_frame, textvariable=self.class_select_var, state="readonly", width=10)
        self.class_select_combo.pack(anchor="w")
        self.class_select_combo.bind("<<ComboboxSelected>>", self.on_class_select_change)
        self.update_class_select_options()
        
        self.class_controls_frame.pack(fill=tk.X)
        
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
        tk.Button(controls_frame, text="Reset", command=self.reset_simulation).pack(anchor="w", pady=10)
        
        # Results Labels
        tk.Label(controls_frame, text="Results:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.accuracy_label = tk.Label(controls_frame, text="Accuracy: N/A")
        self.accuracy_label.pack(anchor="w")
        self.test_samples_label = tk.Label(controls_frame, text="Test Samples: N/A")
        self.test_samples_label.pack(anchor="w")
        self.final_error_label = tk.Label(controls_frame, text="Final Error: N/A")
        self.final_error_label.pack(anchor="w")
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        # Info Display
        self.cycle_label = tk.Label(controls_frame, text="Cycles: 0", font=("Arial", 10))
        self.cycle_label.pack(anchor="w", pady=(10, 0))
        
        self.draw_axes()

    def toggle_mode(self):
        self.mode = self.mode_var.get()
        self.points = []
        self.errors = []
        self.canvas.delete("all")
        
        if self.mode == "Manual":
            self.class_controls_frame.pack(fill=tk.X)
        else:  # Regression
            self.class_controls_frame.pack_forget()
            self.normalize_var.set(True)  # Force normalization for regression
        
        self.draw_axes()
        self.initialize_weights()
    
    def initialize_weights(self):
        # Matrix K x N (Rows x Cols)
        if self.mode == "Regression":
            # For regression: 1 output, 2 inputs (x -> y)
            # Weights: [bias, x]
            self.weights_matrix = [[random.uniform(-0.5, 0.5) for _ in range(2)]]
        else:
            # For classification: K classes, 3 inputs (bias, x, y)
            self.weights_matrix = [[random.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(self.num_classes)]

    def reset_simulation(self):
        self.points = []
        self.errors = []
        self.training_results = []
        self.cycle_count = 0
        self.cycle_label.config(text="Cycles: 0")
        self.initialize_weights()
        self.draw_canvas()
        messagebox.showinfo("Info", "Simulation reset.")

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
        if self.mode == "Regression":
            color = "black"
        else:
            color = self.colors[(label - 1) % len(self.colors)]
        self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, fill=color)

    def draw_canvas(self):
        self.canvas.delete("all")
        self.draw_axes()
        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            self.draw_point(sx, sy, p.label)
        if self.mode == "Manual":  # Only draw decision boundaries in classification mode
            self.draw_decision_boundaries()

    def initialize_randomly(self):
        self.initialize_weights()
        if self.mode == "Regression":
            self.canvas.delete("all")
            self.draw_axes()
            for p in self.points:
                sx, sy = self.cartesian_to_screen(p.x, p.y)
                self.draw_point(sx, sy, 1)
        else:
            self.draw_canvas()

    def draw_decision_boundaries(self):
        if self.mode == "Regression":
            return  # No decision boundaries in regression mode
            
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
            
        if self.mode == "Regression":
            # For regression: normalize x (input) and y (output) separately
            if self.is_normalized:
                xs = [d['x'] for d in data]
                ys = [d['y'] for d in data]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                x_range = x_max - x_min if x_max != x_min else 1.0
                y_range = y_max - y_min if y_max != y_min else 1.0
                
                self.normalization_params_in = (x_min, x_range)
                self.normalization_params_out = (y_min, y_range)
                
                for d in data:
                    d['x'] = (d['x'] - x_min) / x_range
                    d['y'] = (d['y'] - y_min) / y_range
            else:
                self.normalization_params_in = None
                self.normalization_params_out = None
        else:
            # For classification
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
        if self.mode == "Regression":
            messagebox.showinfo("Info", "Discrete training not applicable for regression. Use Continuous (Delta) instead.")
            return
            
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

        # Collect results for regression graph and calculate accuracy
        self.training_results = []
        correct_count = 0
        total_samples = len(training_data)

        for data in training_data:
            inputs = [self.bias, data['x'], data['y']]
            
            # Desired Output Vector d
            d = [-1] * self.num_classes
            d[data['label'] - 1] = 1 
            
            # Net Vector Calculation
            net = []
            for k in range(self.num_classes):
                val = sum(self.weights_matrix[k][j] * inputs[j] for j in range(3))
                net.append(val)
                
            # Output Vector o
            o = [1 if n >= 0 else -1 for n in net]
            
            # Check accuracy (if the max output index matches the label)
            pred_idx = net.index(max(net))
            if pred_idx == (data['label'] - 1):
                correct_count += 1

            # Store each class output vs target
            for k in range(self.num_classes):
                self.training_results.append({'target': d[k], 'output': o[k]})

        accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
        
        # Update GUI labels
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
        self.test_samples_label.config(text=f"Test Samples: {total_samples}")
        self.final_error_label.config(text=f"Final Error: {self.errors[-1] if self.errors else 'N/A'}")

    def train_continuous(self):
        if self.mode == "Regression":
            self.train_regression()
            return
            
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

        # Collect results for regression graph and calculate accuracy
        self.training_results = []
        correct_count = 0
        total_samples = len(training_data)

        for data in training_data:
            inputs = [self.bias, data['x'], data['y']]
            
            # Desired Output Vector d
            d = [0] * self.num_classes
            d[data['label'] - 1] = 1
            
            # Net Vector Calculation
            net = []
            for k in range(self.num_classes):
                val = sum(self.weights_matrix[k][j] * inputs[j] for j in range(3))
                net.append(val)
                
            # Output Vector o
            o = []
            for n in net:
                try:
                    res = 1 / (1 + math.exp(-n))
                except OverflowError:
                    res = 0 if n < 0 else 1
                o.append(res)
            
            # Check accuracy
            pred_idx = o.index(max(o))
            if pred_idx == (data['label'] - 1):
                correct_count += 1

            # Store each class output vs target
            for k in range(self.num_classes):
                self.training_results.append({'target': d[k], 'output': o[k]})

        accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
        
        # Update GUI labels
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
        self.test_samples_label.config(text=f"Test Samples: {total_samples}")
        self.final_error_label.config(text=f"Final Error: {self.errors[-1] if self.errors else 'N/A'}")

    def train_regression(self):
        print("Regression training (Single Layer Linear) selected")
        
        try:
            max_epochs = self.max_epochs_var.get()
            learning_rate = self.learning_rate_var.get()
            min_error = self.min_error_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters.")
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
                # Input: [bias, x]
                inputs = [self.bias, data['x']]
                
                # Desired output: y
                target = data['y']
                
                # Net calculation: net = w0*bias + w1*x
                net = sum(self.weights_matrix[0][j] * inputs[j] for j in range(2))
                
                # Linear output (no activation function)
                output = net
                
                # Error
                error = target - output
                total_error += error ** 2
                
                # Weight update (Delta rule for linear output)
                for j in range(2):
                    self.weights_matrix[0][j] += learning_rate * error * inputs[j]
            
            mse = total_error / len(training_data)
            self.errors.append(mse)
            self.cycle_count += 1
            
            if epoch % 10 == 0:
                self.draw_regression_line()
                self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
                self.root.update()
            
            if mse < min_error:
                print(f"Converged in {epoch+1} epochs. MSE: {mse}")
                break
        
        self.draw_regression_line()
        
        # Calculate final MSE and R²
        total_error = 0
        mean_target = sum([d['y'] for d in training_data]) / len(training_data)
        ss_tot = sum([(d['y'] - mean_target) ** 2 for d in training_data])
        ss_res = 0
        
        for data in training_data:
            inputs = [self.bias, data['x']]
            net = sum(self.weights_matrix[0][j] * inputs[j] for j in range(2))
            output = net
            error = data['y'] - output
            total_error += error ** 2
            ss_res += error ** 2
        
        mse = total_error / len(training_data)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        self.accuracy_label.config(text=f"R²: {r_squared:.4f}")
        self.test_samples_label.config(text=f"Samples: {len(training_data)}")
        self.final_error_label.config(text=f"MSE: {mse:.6f}")

    def draw_regression_line(self):
        self.canvas.delete("all")
        self.draw_axes()
        
        # Draw training points
        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            self.draw_point(sx, sy, 1)
        
        # Draw regression line
        screen_x_coords = range(0, self.canvas_width, 2)
        line_coords = []
        
        for sx in screen_x_coords:
            cx, _ = self.screen_to_cartesian(sx, 0)
            
            # Normalize input if needed
            if self.normalization_params_in:
                x_min, x_range = self.normalization_params_in
                x_norm = (cx - x_min) / x_range
            else:
                x_norm = cx
            
            # Predict
            net = self.weights_matrix[0][0] * self.bias + self.weights_matrix[0][1] * x_norm
            y_pred = net
            
            # Denormalize output if needed
            if self.normalization_params_out:
                y_min, y_range = self.normalization_params_out
                y_real = y_pred * y_range + y_min
            else:
                y_real = y_pred
            
            _, sy = self.cartesian_to_screen(0, y_real)
            line_coords.append((sx, sy))
        
        if len(line_coords) > 1:
            self.canvas.create_line(line_coords, fill="red", width=3)

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
