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
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

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

def mat_mul(A, B): # Element-wise
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
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.prev_weight_updates = []
        self.prev_bias_updates = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            rows, cols = layer_sizes[i], layer_sizes[i+1]
            self.weights.append(mat_rand(rows, cols))
            self.biases.append(mat_rand(1, cols))
            self.prev_weight_updates.append(mat_zeros(rows, cols))
            self.prev_bias_updates.append(mat_zeros(1, cols))

    def forward(self, X):
        self.activations = [X]
        input_data = X
        for w, b in zip(self.weights, self.biases):
            dot_prod = mat_dot(input_data, w)
            # Add bias (broadcast)
            z = []
            for i in range(len(dot_prod)):
                row = []
                for j in range(len(dot_prod[0])):
                    row.append(dot_prod[i][j] + b[0][j])
                z.append(row)
            
            output = mat_sigmoid(z)
            self.activations.append(output)
            input_data = output
        return input_data

    def backward(self, y_true, learning_rate, momentum=0.0):
        output = self.activations[-1]
        error = mat_sub(y_true, output)
        delta = mat_mul(error, mat_sigmoid_derivative(output))
        
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta_prev = deltas[-1]
            w_next = self.weights[i]
            activation = self.activations[i]
            
            error_hidden = mat_dot(delta_prev, mat_transpose(w_next))
            delta_hidden = mat_mul(error_hidden, mat_sigmoid_derivative(activation))
            deltas.append(delta_hidden)
            
        deltas.reverse()
        
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
        
        # MSE
        total_error = 0
        count = 0
        for r in error:
            for val in r:
                total_error += val ** 2
                count += 1
        return total_error / count if count > 0 else 0

@dataclass

class Layer:
	neurons: int
	activation: str

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
        self.nn = None
        self.normalization_params = None
        self.error_history = []
        self.training_results = []
        
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
        self.neurons_per_layer_var = tk.StringVar(value="5")
        tk.Entry(controls_frame, textvariable=self.neurons_per_layer_var).pack(anchor="w", pady=(0, 10))

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
        tk.Button(controls_frame, text="Clear Points", command=self.clear_points).pack(anchor="w", pady=5)

        # Draw Axes
        self.draw_axes()
        
        # Setup Menu
        self.setup_menu()

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
        self.points.append(Point(cx, cy, self.current_class))
        self.draw_point(event.x, event.y, self.current_class)
        
    def draw_point(self, x, y, label):
        r = 4
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
            
            # If user specified more layers but only one size, replicate it
            if len(neurons_per_layer) == 1 and num_layers > 1:
                neurons_per_layer = neurons_per_layer * num_layers
            
            print("hidden layers:", neurons_per_layer)
            
            input_size = 2
            output_size = self.num_classes
            
            self.nn = MultiLayerPerceptron(input_size, neurons_per_layer, output_size)
            print("Weights initialized.")
            print(f"Structure: Input({input_size}) -> Hidden({neurons_per_layer}) -> Output({output_size})")
            
        except ValueError:
            print("Invalid hidden layers configuration.")

    def get_training_data(self):
        if not self.points:
            return None, None
            
        X = [[p.x, p.y] for p in self.points]
        y = [[0.0] * self.num_classes for _ in range(len(self.points))]
        for i, p in enumerate(self.points):
            y[i][p.label - 1] = 1.0
            
        if self.normalize_var.get():
            # Find min/max for each col
            min_vals = [float('inf'), float('inf')]
            max_vals = [float('-inf'), float('-inf')]
            
            for row in X:
                for j in range(2):
                    if row[j] < min_vals[j]: min_vals[j] = row[j]
                    if row[j] > max_vals[j]: max_vals[j] = row[j]
            
            range_vals = [max_vals[j] - min_vals[j] for j in range(2)]
            for j in range(2):
                if range_vals[j] == 0: range_vals[j] = 1.0
            
            self.normalization_params = (min_vals, range_vals)
            
            # Normalize
            X_norm = []
            for row in X:
                new_row = []
                for j in range(2):
                    new_row.append((row[j] - min_vals[j]) / range_vals[j])
                X_norm.append(new_row)
            X = X_norm
        else:
            self.normalization_params = None
            
        return X, y

    def train(self, momentum=0.0):
        if not self.points:
            print("No points to train on.")
            return

        if self.nn is None:
            self.initialize_weights()
            
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
                
            if mse < min_error:
                print(f"Converged at epoch {epoch}, Error: {mse}")
                break
        
        print(f"Training finished. Final Error: {self.error_history[-1] if self.error_history else 'N/A'}")
        
        # Collect results for regression graph
        self.training_results = []
        final_output = self.nn.forward(X)
        for i in range(len(X)):
            for j in range(len(final_output[i])):
                self.training_results.append({
                    'target': y[i][j],
                    'output': final_output[i][j]
                })
                
        self.update_main_regression()

    def train_with_momentum(self):
        print("Training with momentum...")
        self.train(momentum=0.9)

    def train_without_momentum(self):
        print("Training without momentum...")
        self.train(momentum=0.0)

    def test_network(self):
        if self.nn is None:
            print("Network not initialized.")
            return
            
        X, y = self.get_training_data()
        if X is None: return
        
        output = self.nn.forward(X)
        
        correct = 0
        for i in range(len(output)):
            pred_idx = output[i].index(max(output[i]))
            true_idx = y[i].index(max(y[i]))
            if pred_idx == true_idx:
                correct += 1
        
        accuracy = correct / len(output)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def show_error_graph(self):
        if not self.error_history:
            print("No training history available.")
            return
            
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Error Graph")
        graph_window.geometry("600x400")
        
        canvas = tk.Canvas(graph_window, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        w = 600
        h = 400
        padding = 50
        
        max_error = max(self.error_history)
        num_epochs = len(self.error_history)
        
        # Draw axes
        canvas.create_line(padding, h - padding, w - padding, h - padding, arrow=tk.LAST) # X axis
        canvas.create_line(padding, h - padding, padding, padding, arrow=tk.LAST) # Y axis
        
        # Labels
        canvas.create_text(w/2, h - 15, text="Epochs", font=("Arial", 10))
        canvas.create_text(15, h/2, text="Error (MSE)", angle=90, font=("Arial", 10))
        
        if num_epochs < 2:
            return

        x_scale = (w - 2 * padding) / (num_epochs - 1)
        y_scale = (h - 2 * padding) / (max_error if max_error > 0 else 1)
        
        points = []
        for i, error in enumerate(self.error_history):
            x = padding + i * x_scale
            y = h - padding - error * y_scale
            points.append((x, y))
            
        canvas.create_line(points, fill="blue", width=2)
        
        # Ticks
        # Y-axis max
        canvas.create_text(padding - 5, h - padding - max_error * y_scale, text=f"{max_error:.4f}", anchor="e", font=("Arial", 8))
        canvas.create_line(padding - 2, h - padding - max_error * y_scale, padding + 2, h - padding - max_error * y_scale)
        
        # Y-axis zero
        canvas.create_text(padding - 5, h - padding, text="0.0000", anchor="e", font=("Arial", 8))
        
        # X-axis max
        canvas.create_text(w - padding, h - padding + 5, text=f"{num_epochs}", anchor="n", font=("Arial", 8))

    def draw_regression_on_canvas(self, canvas):
        res = 5 # Resolution in pixels
        width = self.canvas_width
        height = self.canvas_height
        
        coords = []
        pixels = []
        
        for y in range(0, height, res):
            for x in range(0, width, res):
                cx, cy = self.screen_to_cartesian(x, y)
                coords.append([cx, cy])
                pixels.append((x, y))
                
        X_grid = coords
        
        if self.normalization_params:
            min_vals, range_vals = self.normalization_params
            X_norm = []
            for row in X_grid:
                new_row = []
                for j in range(2):
                    new_row.append((row[j] - min_vals[j]) / range_vals[j])
                X_norm.append(new_row)
            X_grid = X_norm
            
        output = self.nn.forward(X_grid)
        
        predictions = []
        for row in output:
            predictions.append(row.index(max(row)) + 1)
        
        for (x, y), pred in zip(pixels, predictions):
            color = self.colors[(pred - 1) % len(self.colors)]
            canvas.create_rectangle(x, y, x+res, y+res, fill=color, outline="", stipple="gray25")
            
        # Draw axes
        cw = self.canvas_width
        ch = self.canvas_height
        canvas.create_line(0, ch/2, cw, ch/2, fill="black", width=2)
        canvas.create_text(cw - 20, ch/2 + 20, text="x1", font=("Arial", 12, "bold"))
        canvas.create_line(cw/2, 0, cw/2, ch, fill="black", width=2)
        canvas.create_text(cw/2 + 20, 20, text="x2", font=("Arial", 12, "bold"))

        for p in self.points:
            sx, sy = self.cartesian_to_screen(p.x, p.y)
            r = 4
            color = self.colors[(p.label - 1) % len(self.colors)]
            canvas.create_oval(sx-r, sy-r, sx+r, sy+r, outline="black", fill=color)

    def show_regression_graph(self):
        if not self.training_results:
            messagebox.showinfo("Info", "No training results to display.")
            return
            
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Regression Graph (Target vs Output)")
        graph_window.geometry("600x400")
        
        canvas = tk.Canvas(graph_window, bg="white", width=550, height=350)
        canvas.pack(padx=20, pady=20)
        
        targets = [d['target'] for d in self.training_results]
        outputs = [d['output'] for d in self.training_results]
        
        min_val = min(min(targets), min(outputs))
        max_val = max(max(targets), max(outputs))
        
        # Add some padding
        padding = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
        min_val -= padding
        max_val += padding
        val_range = max_val - min_val
        
        # Map value to screen
        def val_to_screen(v, is_x):
            norm = (v - min_val) / val_range
            if is_x:
                return 50 + norm * 450
            else:
                return 300 - norm * 250
                
        # Draw Axes Lines
        zero_x = val_to_screen(0, True)
        zero_y = val_to_screen(0, False)
        
        axis_x = zero_x if 50 <= zero_x <= 500 else 50
        axis_y = zero_y if 50 <= zero_y <= 300 else 300
        
        canvas.create_line(50, axis_y, 500, axis_y, width=1, fill="gray")
        canvas.create_line(axis_x, 50, axis_x, 300, width=1, fill="gray")
        
        # Labels
        canvas.create_text(275, 330, text="Target Output")
        canvas.create_text(20, 175, text="Actual Output", angle=90)
        
        # Draw Ideal Line (y=x)
        p1_x, p1_y = val_to_screen(min_val, True), val_to_screen(min_val, False)
        p2_x, p2_y = val_to_screen(max_val, True), val_to_screen(max_val, False)
        canvas.create_line(p1_x, p1_y, p2_x, p2_y, fill="green", dash=(4, 4))
        
        # Plot Points
        for res in self.training_results:
            x = val_to_screen(res['target'], True)
            y = val_to_screen(res['output'], False)
            canvas.create_oval(x-3, y-3, x+3, y+3, fill="blue", outline="blue")

    def update_main_regression(self):
        if self.nn is None:
            return
        self.canvas.delete("all")
        self.draw_regression_on_canvas(self.canvas)
        
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
