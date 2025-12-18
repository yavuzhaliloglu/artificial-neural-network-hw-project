import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from dataclasses import dataclass
import math
import random

# --- Matrix Math Helpers ---
def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_rand(rows, cols):
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)] # Weights slightly smaller for stability

def mat_dot(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B: raise ValueError("Shape mismatch") 
    C = mat_zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            sum_val = 0.0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
    return C

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_mul(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_sum_cols(A):
    rows, cols = len(A), len(A[0])
    res = [[0.0] * cols]
    for j in range(cols):
        s = 0.0
        for i in range(rows):
            s += A[i][j]
        res[0][j] = s
    return res

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def mat_sigmoid(A):
    return [[sigmoid(A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]

def mat_sigmoid_derivative(A):
    # x * (1 - x)
    return [[A[i][j] * (1.0 - A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]
# ---------------------------

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layers, output_size, is_regression=False):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.is_regression = is_regression # NEW: Mode flag
        self.weights = []
        self.biases = []
        self.prev_weight_updates = []
        self.prev_bias_updates = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        print(f"Network Structure: {layer_sizes} (Regression: {self.is_regression})")
        
        for i in range(len(layer_sizes) - 1):
            rows, cols = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(mat_rand(rows, cols))
            self.biases.append(mat_rand(1, cols))
            self.prev_weight_updates.append(mat_zeros(rows, cols))
            self.prev_bias_updates.append(mat_zeros(1, cols))

    def forward(self, X):
        self.activations = [X]
        input_data = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dot_prod = mat_dot(input_data, w)
            # Add bias (broadcast)
            z = []
            for r_idx in range(len(dot_prod)):
                row = []
                for c_idx in range(len(dot_prod[0])):
                    row.append(dot_prod[r_idx][c_idx] + b[0][c_idx])
                z.append(row)
            
            # Check if it is the last layer AND we are in regression mode
            is_last_layer = (i == len(self.weights) - 1)
            
            if self.is_regression and is_last_layer:
                # Linear Activation (Identity) for Regression Output
                output = z 
            else:
                # Sigmoid Activation for Hidden Layers or Classification Output
                output = mat_sigmoid(z)
                
            self.activations.append(output)
            input_data = output
        return input_data

    def backward(self, y_true, learning_rate, momentum=0.0):
        output = self.activations[-1]
        error = mat_sub(y_true, output)
        
        # Calculate Delta for Output Layer
        if self.is_regression:
            # Derivative of Linear function (f(x)=x) is 1.
            # So delta is just the error.
            delta = error
        else:
            # Derivative of Sigmoid
            delta = mat_mul(error, mat_sigmoid_derivative(output))
        
        deltas = [delta]
        
        # Hidden Layers (Backpropagate)
        for i in range(len(self.weights) - 1, 0, -1):
            delta_prev = deltas[-1]
            w_next = self.weights[i]
            activation = self.activations[i]
            
            error_hidden = mat_dot(delta_prev, mat_transpose(w_next))
            delta_hidden = mat_mul(error_hidden, mat_sigmoid_derivative(activation))
            deltas.append(delta_hidden)
            
        deltas.reverse()
        
        # Update Weights
        for i in range(len(self.weights)):
            input_act = self.activations[i]
            delta = deltas[i]
            
            grad_w = mat_dot(mat_transpose(input_act), delta)
            weight_update = mat_scale(grad_w, learning_rate)
            
            grad_b = mat_sum_cols(delta)
            bias_update = mat_scale(grad_b, learning_rate)
            
            term_w = mat_scale(self.prev_weight_updates[i], momentum)
            weight_update = mat_add(weight_update, term_w)
            
            term_b = mat_scale(self.prev_bias_updates[i], momentum)
            bias_update = mat_add(bias_update, term_b)
            
            self.weights[i] = mat_add(self.weights[i], weight_update)
            self.biases[i] = mat_add(self.biases[i], bias_update)
            
            self.prev_weight_updates[i] = weight_update
            self.prev_bias_updates[i] = bias_update
        
        # MSE Calculation
        total_error = 0
        count = 0
        for r in error:
            for val in r:
                total_error += val ** 2
                count += 1
        return total_error / count if count > 0 else 0

@dataclass
class Point:
    x: float
    y: float
    label: int

class MultiLayerNNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Layer Neural Network (Classification & Regression)")
        self.root.geometry("1200x800")
        
        self.points = []
        self.num_classes = 2
        self.current_class = 1
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        self.nn = None
        self.normalization_params_in = None
        self.normalization_params_out = None
        self.error_history = []
        
        # Main Layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Controls Frame
        self.controls_frame = tk.Frame(main_frame)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, anchor="n")
        
        # Mode Selection
        tk.Label(self.controls_frame, text="Mode:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.mode_var = tk.StringVar(value="Manual")
        
        tk.Radiobutton(self.controls_frame, text="Classification (Manual)", variable=self.mode_var, value="Manual", command=self.toggle_mode).pack(anchor="w")
        tk.Radiobutton(self.controls_frame, text="Regression (Curve Fitting)", variable=self.mode_var, value="Regression", command=self.toggle_mode).pack(anchor="w")
        
        tk.Label(self.controls_frame, text="").pack() # Spacer

        # --- Manual/Regression Frame ---
        self.manual_frame = tk.Frame(self.controls_frame)
        
        # Classification Specifics
        self.class_controls_frame = tk.Frame(self.manual_frame)
        tk.Label(self.class_controls_frame, text="Class Count:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.class_count_var = tk.IntVar(value=2)
        self.class_count_combo = ttk.Combobox(self.class_controls_frame, textvariable=self.class_count_var, values=[2, 3, 4, 5, 6], state="readonly", width=5)
        self.class_count_combo.pack(anchor="w")
        self.class_count_combo.bind("<<ComboboxSelected>>", self.on_class_count_change)
        
        tk.Label(self.class_controls_frame, text="Select Class:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(5, 5))
        self.class_select_var = tk.StringVar(value="Class 1")
        self.class_select_combo = ttk.Combobox(self.class_controls_frame, textvariable=self.class_select_var, state="readonly", width=10)
        self.class_select_combo.pack(anchor="w")
        self.class_select_combo.bind("<<ComboboxSelected>>", self.on_class_select_change)
        self.class_controls_frame.pack(fill=tk.X)
        self.update_class_select_options()
        
        tk.Button(self.manual_frame, text="Clear Points", command=self.clear_points).pack(anchor="w", pady=15)

        # --- Common Controls ---
        self.common_frame = tk.Frame(self.controls_frame)
        
        tk.Label(self.common_frame, text="Hidden Layers (Count):").pack(anchor="w", pady=(10, 5))
        self.num_layers_var = tk.IntVar(value=2)
        tk.Entry(self.common_frame, textvariable=self.num_layers_var).pack(anchor="w", pady=(0, 10))

        tk.Label(self.common_frame, text="Neurons per Layer (e.g. 5,5):").pack(anchor="w", pady=(0, 5))
        self.neurons_per_layer_var = tk.StringVar(value="8,8")
        tk.Entry(self.common_frame, textvariable=self.neurons_per_layer_var).pack(anchor="w", pady=(0, 10))

        # Normalize Checkbox
        self.normalize_var = tk.BooleanVar(value=True)
        self.normalize_check = tk.Checkbutton(self.common_frame, text="Normalize Data", variable=self.normalize_var)
        self.normalize_check.pack(anchor="w", pady=(0, 10))

        tk.Label(self.common_frame, text="Max Epochs:").pack(anchor="w", pady=(0, 5))
        self.max_epochs_var = tk.IntVar(value=2000)
        tk.Entry(self.common_frame, textvariable=self.max_epochs_var).pack(anchor="w", pady=(0, 10))

        tk.Label(self.common_frame, text="Learning Rate:").pack(anchor="w", pady=(0, 5))
        self.learning_rate_var = tk.DoubleVar(value=0.05)
        tk.Entry(self.common_frame, textvariable=self.learning_rate_var).pack(anchor="w", pady=(0, 10))

        tk.Label(self.common_frame, text="Min Error:").pack(anchor="w", pady=(0, 5))
        self.min_error_var = tk.DoubleVar(value=0.001)
        tk.Entry(self.common_frame, textvariable=self.min_error_var).pack(anchor="w", pady=(0, 10))

        tk.Button(self.common_frame, text="Show Error Graph", command=self.show_error_graph).pack(anchor="w", pady=5)

        # Results Labels
        tk.Label(self.common_frame, text="Results:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.accuracy_label = tk.Label(self.common_frame, text="Accuracy: N/A")
        self.accuracy_label.pack(anchor="w")
        self.final_error_label = tk.Label(self.common_frame, text="Final Error: N/A")
        self.final_error_label.pack(anchor="w")

        # Initial Pack
        self.manual_frame.pack(fill=tk.X)
        self.common_frame.pack(fill=tk.X)
        
        self.draw_axes()
        self.setup_menu()

    def toggle_mode(self):
        mode = self.mode_var.get()
        
        # Reset frames
        self.manual_frame.pack_forget()
        self.class_controls_frame.pack_forget()
        
        self.canvas.delete("all")
        self.points = []
        
        if mode == "Manual": # Classification Manual
            self.manual_frame.pack(fill=tk.X, before=self.common_frame)
            self.class_controls_frame.pack(fill=tk.X) # Show class controls
            self.normalize_check.config(state="normal")
            self.draw_axes()
            
        elif mode == "Regression": # Regression Manual
            self.manual_frame.pack(fill=tk.X, before=self.common_frame)
            # Hide class controls (regression doesn't have classes)
            self.normalize_check.config(state="normal")
            # Force normalization on for regression usually
            self.normalize_var.set(True) 
            self.draw_axes()

    def clear_points(self):
        self.points = []
        self.draw_canvas()

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
        
        # In Regression mode, label doesn't really matter, use 1
        label = self.current_class if self.mode_var.get() == "Manual" else 1
        
        self.points.append(Point(cx, cy, label))
        self.draw_point(event.x, event.y, label)
        
    def draw_point(self, x, y, label):
        r = 4
        if self.mode_var.get() == "Regression":
            color = "black"
        else:
            color = self.colors[(label - 1) % len(self.colors)]
        self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="black", fill=color)

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
        try:
            num_layers = self.num_layers_var.get()
            neurons_per_layer_var = self.neurons_per_layer_var.get()
            neurons_per_layer = [int(x.strip()) for x in neurons_per_layer_var.split(',') if x.strip()]
            
            if len(neurons_per_layer) == 1 and num_layers > 1:
                neurons_per_layer = neurons_per_layer * num_layers
            
            mode = self.mode_var.get()
            is_regression = (mode == "Regression")
            
            if mode == "Manual": # Classification 2D
                input_size = 2
                output_size = self.num_classes
            elif mode == "Regression": # Regression 1D (x -> y)
                input_size = 1
                output_size = 1
            
            self.nn = MultiLayerPerceptron(input_size, neurons_per_layer, output_size, is_regression=is_regression)
            print("Weights initialized.")
            
        except ValueError:
            print("Invalid hidden layers configuration.")

    def normalize_matrix(self, data):
        # Data is list of lists
        cols = len(data[0])
        min_vals = [float('inf')] * cols
        max_vals = [float('-inf')] * cols
        
        for row in data:
            for j in range(cols):
                if row[j] < min_vals[j]: min_vals[j] = row[j]
                if row[j] > max_vals[j]: max_vals[j] = row[j]
        
        range_vals = []
        for j in range(cols):
            r = max_vals[j] - min_vals[j]
            if r == 0: r = 1.0
            range_vals.append(r)
            
        normalized_data = []
        for row in data:
            new_row = []
            for j in range(cols):
                new_row.append((row[j] - min_vals[j]) / range_vals[j])
            normalized_data.append(new_row)
            
        return normalized_data, min_vals, range_vals

    def get_training_data(self):
        mode = self.mode_var.get()
        
        # Manual or Regression
        if not self.points: return None, None
            
        if mode == "Manual":
            # Input: (x, y), Output: One-hot Class
            X = [[p.x, p.y] for p in self.points]
            y = [[0.0] * self.num_classes for _ in range(len(self.points))]
            for i, p in enumerate(self.points): y[i][p.label - 1] = 1.0
        
        elif mode == "Regression":
            # Input: (x), Output: (y)
            X = [[p.x] for p in self.points]
            y = [[p.y] for p in self.points]
            
        # Normalization
        if self.normalize_var.get():
            X, min_in, range_in = self.normalize_matrix(X)
            self.normalization_params_in = (min_in, range_in)
            
            # Also normalize targets for regression
            if mode == "Regression":
                y, min_out, range_out = self.normalize_matrix(y)
                self.normalization_params_out = (min_out, range_out)
            else:
                self.normalization_params_out = None
        else:
            self.normalization_params_in = None
            self.normalization_params_out = None
            
        return X, y

    def train(self, momentum=0.0):
        if self.nn is None: self.initialize_weights()
            
        X, y = self.get_training_data()
        if X is None: return
        
        epochs = self.max_epochs_var.get()
        learning_rate = self.learning_rate_var.get()
        min_error = self.min_error_var.get()
        self.error_history = []
        
        for epoch in range(epochs):
            self.nn.forward(X)
            mse = self.nn.backward(y, learning_rate, momentum)
            self.error_history.append(mse)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {mse}")
                self.root.update()
                
            if mse < min_error: break
        
        print(f"Final Error: {self.error_history[-1]}")
        
        # Calculate Accuracy / Update UI
        if self.mode_var.get() == "Regression":
            self.accuracy_label.config(text="Accuracy: N/A (Regression)")
            self.update_main_regression() # Draw the line
        else:
            final_output = self.nn.forward(X)
            correct_count = 0
            for i in range(len(X)):
                pred_idx = final_output[i].index(max(final_output[i]))
                true_idx = y[i].index(max(y[i]))
                if pred_idx == true_idx: correct_count += 1
            accuracy = (correct_count / len(X)) * 100
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
            if self.mode_var.get() == "Manual":
                self.update_main_classification()

        self.final_error_label.config(text=f"Final Error: {self.error_history[-1]:.6f}")

    def train_with_momentum(self): self.train(momentum=0.9)
    def train_without_momentum(self): self.train(momentum=0.0)

    def test_network(self):
        # Simplified test
        print("Test triggered.")

    def show_error_graph(self):
        if not self.error_history: return
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Error Graph")
        graph_window.geometry("600x400")
        canvas = tk.Canvas(graph_window, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        w, h, padding = 600, 400, 50
        max_error = max(self.error_history)
        num_epochs = len(self.error_history)
        
        canvas.create_line(padding, h - padding, w - padding, h - padding, arrow=tk.LAST)
        canvas.create_line(padding, h - padding, padding, padding, arrow=tk.LAST)
        
        if num_epochs < 2: return
        x_scale = (w - 2 * padding) / (num_epochs - 1)
        y_scale = (h - 2 * padding) / (max_error if max_error > 0 else 1)
        
        points = []
        for i, error in enumerate(self.error_history):
            x = padding + i * x_scale
            y = h - padding - error * y_scale
            points.append((x, y))
        canvas.create_line(points, fill="blue", width=2)

    def update_main_classification(self):
        self.canvas.delete("all")
        
        # Grid classification drawing
        res = 6
        coords, pixels = [], []
        for y in range(0, self.canvas_height, res):
            for x in range(0, self.canvas_width, res):
                cx, cy = self.screen_to_cartesian(x, y)
                coords.append([cx, cy])
                pixels.append((x, y))
        
        X_grid = coords
        if self.normalization_params_in:
            min_v, range_v = self.normalization_params_in
            X_norm = []
            for row in X_grid:
                X_norm.append([(row[0]-min_v[0])/range_v[0], (row[1]-min_v[1])/range_v[1]])
            X_grid = X_norm
            
        output = self.nn.forward(X_grid)
        
        for (x, y), row in zip(pixels, output):
            pred = row.index(max(row)) + 1
            color = self.colors[(pred - 1) % len(self.colors)]
            self.canvas.create_rectangle(x, y, x+res, y+res, fill=color, outline="", stipple="gray25")
            
        self.draw_axes()
        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            self.draw_point(sx, sy, p.label)

    def update_main_regression(self):
        self.canvas.delete("all")
        self.draw_axes()
        
        # 1. Draw original points
        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            self.draw_point(sx, sy, 1) # Label 1 for black dots
            
        # 2. Draw Regression Line
        # Iterate across X screen coordinates
        screen_x_coords = range(0, self.canvas_width, 2) # Step 2 pixels
        
        # Prepare input batch
        X_batch = []
        for sx in screen_x_coords:
            cx, _ = self.screen_to_cartesian(sx, 0) # y doesn't matter for input
            X_batch.append([cx])
            
        # Normalize Input
        if self.normalization_params_in:
            min_in, range_in = self.normalization_params_in
            X_norm = []
            for row in X_batch:
                X_norm.append([(row[0] - min_in[0]) / range_in[0]])
            X_batch_ready = X_norm
        else:
            X_batch_ready = X_batch
            
        # Predict
        Y_pred_batch = self.nn.forward(X_batch_ready)
        
        # Denormalize Output
        final_coords = []
        if self.normalization_params_out:
            min_out, range_out = self.normalization_params_out
            for i, val_list in enumerate(Y_pred_batch):
                y_val_norm = val_list[0]
                y_val_real = (y_val_norm * range_out[0]) + min_out[0]
                
                # Convert back to screen
                sx = screen_x_coords[i]
                _, sy = self.cartesian_to_screen(0, y_val_real) # 0 for x placeholder
                final_coords.append((sx, sy))
        else:
            for i, val_list in enumerate(Y_pred_batch):
                sx = screen_x_coords[i]
                _, sy = self.cartesian_to_screen(0, val_list[0])
                final_coords.append((sx, sy))
                
        # Draw the line
        if len(final_coords) > 1:
            self.canvas.create_line(final_coords, fill="red", width=3)

    def draw_axes(self):
        cw, ch = self.canvas_width, self.canvas_height
        self.canvas.create_line(0, ch/2, cw, ch/2, fill="black", width=2)
        self.canvas.create_text(cw - 20, ch/2 + 20, text="x", font=("Arial", 12, "bold"))
        self.canvas.create_line(cw/2, 0, cw/2, ch, fill="black", width=2)
        self.canvas.create_text(cw/2 + 20, 20, text="y", font=("Arial", 12, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiLayerNNGUI(root)
    root.mainloop()