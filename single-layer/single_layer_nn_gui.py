import tkinter as tk
from tkinter import simpledialog, messagebox
import random
import time
import math
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    label: int

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Yapay Sinir Ağları - Ödev")
        self.root.geometry("800x600")
        
        # Data
        self.points = [] # List of {'x': float, 'y': float, 'label': int}
        self.weights = [0.0, 0.0, 0.0] # w0 (bias), w1, w2
        self.learning_rate = 1
        self.bias = 1 # Bias input value
        self.errors = []
        self.training_results = []
        
        self.is_normalized = False
        self.norm_params = {}
        
        # UI Layout
        self.setup_ui()
        
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
        init_menu.add_command(label="Manually", command=self.initialize_manually)
        
        process_menu.add_separator()
        
        # Training Submenu
        train_menu = tk.Menu(process_menu, tearoff=0)
        process_menu.add_cascade(label="Training", menu=train_menu)
        train_menu.add_command(label="Binary", command=self.train_binary)
        train_menu.add_command(label="Continuous", command=self.train_continuous)
        
        process_menu.add_separator()
        process_menu.add_command(label="Exit", command=self.root.quit)
        
        # Main Layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas (Left)
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Controls (Right)
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10, anchor="n")
        
        # Class Selection
        self.class_var = tk.IntVar(value=1)
        tk.Label(controls_frame, text="Select Class:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        tk.Radiobutton(controls_frame, text="Class 1", variable=self.class_var, value=1).pack(anchor="w")
        tk.Radiobutton(controls_frame, text="Class 2", variable=self.class_var, value=2).pack(anchor="w")
        
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
        tk.Button(controls_frame, text="Show Regression Graph", command=self.show_regression_graph).pack(anchor="w", pady=0)
        tk.Button(controls_frame, text="Reset", command=self.reset_simulation).pack(anchor="w", pady=10)
        
        tk.Label(controls_frame, text="").pack() # Spacer
        
        # Weights Display
        self.weights_label = tk.Label(controls_frame, text="0.000 (w1) x1 + 0.000 (w2) x2 + 0.000 (w0) = 0", font=("Arial", 8))
        self.weights_label.pack(anchor="w")

        self.cycle_label = tk.Label(controls_frame, text="Cycles: 0", font=("Arial", 10))
        self.cycle_label.pack(anchor="w", pady=(10, 0))
        
        self.draw_axes()

    def draw_axes(self):
        cw = self.canvas_width
        ch = self.canvas_height
        self.canvas.create_line(cw/2, 0, cw/2, ch, fill="black", width=2) # x2 axis
        self.canvas.create_line(0, ch/2, cw, ch/2, fill="black", width=2) # x1 axis
        
        self.canvas.create_text(cw - 20, ch/2 + 20, text="x1", font=("Arial", 12, "bold"))
        self.canvas.create_text(cw/2 + 20, 20, text="x2", font=("Arial", 12, "bold"))
        
    def screen_to_cartesian(self, sx, sy):
        # Map screen coordinates to cartesian (e.g., -10 to 10)
        # Center is (250, 250) -> (0, 0)
        # Scale: let's say 500 pixels = 20 units (-10 to 10)
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
        label = self.class_var.get()
        # Class 1 is usually -1 or 0, Class 2 is 1. Or 1 and -1.
        # Let's store the raw label from radiobutton (1 or 2) and handle logic later.
        self.points.append(Point(cx, cy, label))
        self.draw_point(event.x, event.y, label)
        
    def draw_point(self, x, y, label):
        r = 4
        if label == 1:
            # Draw 'x'
            self.canvas.create_line(x-r, y-r, x+r, y+r, fill="black", width=2)
            self.canvas.create_line(x-r, y+r, x+r, y-r, fill="black", width=2)
        else:
            # Draw red circle
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline="red", fill="red")

    def initialize_randomly(self):
        self.is_normalized = False
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.update_weights_display()
        self.draw_decision_boundary()

    def initialize_manually(self):
        self.is_normalized = False
        try:
            w0 = simpledialog.askfloat("Input", "Enter w0 (bias):", parent=self.root)
            if w0 is None: return
            w1 = simpledialog.askfloat("Input", "Enter w1:", parent=self.root)
            if w1 is None: return
            w2 = simpledialog.askfloat("Input", "Enter w2:", parent=self.root)
            if w2 is None: return
            
            self.weights = [w0, w1, w2]
            self.update_weights_display()
            self.draw_decision_boundary()
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numbers.")

    def reset_simulation(self):
        self.points = []
        self.errors = []
        self.training_results = []
        self.cycle_count = 0
        self.cycle_label.config(text="Cycles: 0")
        self.initialize_randomly()
        self.canvas.delete("all")
        self.draw_axes()
        messagebox.showinfo("Info", "Simulation reset.")

    def update_weights_display(self):
        w0, w1, w2 = self.weights
        self.weights_label.config(text=f"{w1:.3f} (w1) x1 + {w2:.3f} (w2) x2 + {w0:.3f} (w0) = 0")

    def draw_decision_boundary(self):
        # Clear previous lines (tag 'boundary')
        self.canvas.delete("boundary")
        
        w0, w1, w2 = self.weights
        
        if w2 == 0:
            if w1 == 0: return # No line
            
            if self.is_normalized and self.norm_params:
                # x_train = -w0*bias/w1
                x_train = -w0 * self.bias / w1
                # Convert back to screen coordinates
                # x_train = 2*(x - x_min)/x_range - 1  =>  x = (x_train + 1)*x_range/2 + x_min
                x_val = (x_train + 1) * self.norm_params['x_range'] / 2 + self.norm_params['x_min']
            else:
                # x = -w0*bias/w1 (Vertical line)
                x_val = -w0 * self.bias / w1
                
            sx1, sy1 = self.cartesian_to_screen(x_val, 10)
            sx2, sy2 = self.cartesian_to_screen(x_val, -10)
        else:
            # Calculate y for x = -10 and x = 10 (our logical bounds)
            x1_bound, x2_bound = -10, 10
            
            if self.is_normalized and self.norm_params:
                # We need to find y for x1_bound and x2_bound
                # First convert bounds to normalized space
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
            
        self.canvas.create_line(sx1, sy1, sx2, sy2, fill="blue", width=2, tags="boundary")

    def train_binary(self):
        print("Binary training (Perceptron) selected")
        
        try:
            max_epochs = self.max_epochs_var.get()
            learning_rate = self.learning_rate_var.get()
            min_error = self.min_error_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check Learning Rate, Max Epochs, and Min Error.")
            return

        self.cycle_count = 0
        self.errors = []
        
        # Check normalization
        self.is_normalized = self.normalize_var.get()
        self.norm_params = {}
        
        # Map labels: Class 1 -> 1, Class 2 -> -1
        training_data = []
        for p in self.points:
            target = 1 if p.label == 1 else -1
            training_data.append({'x': p.x, 'y': p.y, 'target': target})
            
        if not training_data:
            messagebox.showwarning("Warning", "No data points to train!")
            return

        if self.is_normalized:
            xs = [d['x'] for d in training_data]
            ys = [d['y'] for d in training_data]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0
            
            self.norm_params = {
                'x_min': x_min, 'x_range': x_range,
                'y_min': y_min, 'y_range': y_range
            }
            
            # Normalize to [-1, 1]
            for d in training_data:
                d['x'] = 2 * (d['x'] - x_min) / x_range - 1
                d['y'] = 2 * (d['y'] - y_min) / y_range - 1

        for epoch in range(max_epochs):
            error_count = 0
            for data in training_data:
                # Calculate Net Input
                # weights: w0(bias), w1(x), w2(y)
                net = self.weights[0] * self.bias + self.weights[1] * data['x'] + self.weights[2] * data['y']
                
                # Activation (Sign function)
                output = 1 if net >= 0 else -1
                
                # Error
                error = data['target'] - output
                
                if error != 0:
                    error_count += 1
                    # Update weights
                    # w_new = w_old + learning_rate * error * input
                    self.weights[0] += learning_rate * error * self.bias
                    self.weights[1] += learning_rate * error * data['x']
                    self.weights[2] += learning_rate * error * data['y']
            
            self.errors.append(error_count)
            self.cycle_count += 1
            self.update_weights_display()
            self.draw_decision_boundary()
            self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
            self.root.update()
            # time.sleep(0.010)
            
            if error_count == 0:
                print(f"Converged in {epoch+1} epochs.")
                break
        else:
             print("Did not converge within max epochs.")

        # Collect results for regression graph
        self.training_results = []
        for data in training_data:
            net = self.weights[0] * self.bias + self.weights[1] * data['x'] + self.weights[2] * data['y']
            output = 1 if net >= 0 else -1
            self.training_results.append({'target': data['target'], 'output': output})

    def train_continuous(self):
        print("Continuous training (Delta) selected")
        
        try:
            max_epochs = self.max_epochs_var.get()
            learning_rate = self.learning_rate_var.get()
            min_error = self.min_error_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters. Please check Learning Rate, Max Epochs, and Min Error.")
            return

        self.cycle_count = 0
        self.errors = []
        
        # Check normalization
        self.is_normalized = self.normalize_var.get()
        self.norm_params = {}
        
        # Map labels: Class 1 -> 1, Class 2 -> 0 (for Sigmoid)
        training_data = []
        for p in self.points:
            target = 1 if p.label == 1 else 0
            training_data.append({'x': p.x, 'y': p.y, 'target': target})
            
        if not training_data:
            messagebox.showwarning("Warning", "No data points to train!")
            return

        if self.is_normalized:
            xs = [d['x'] for d in training_data]
            ys = [d['y'] for d in training_data]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            x_range = x_max - x_min if x_max != x_min else 1.0
            y_range = y_max - y_min if y_max != y_min else 1.0
            
            self.norm_params = {
                'x_min': x_min, 'x_range': x_range,
                'y_min': y_min, 'y_range': y_range
            }
            
            # Normalize to [-1, 1]
            for d in training_data:
                d['x'] = 2 * (d['x'] - x_min) / x_range - 1
                d['y'] = 2 * (d['y'] - y_min) / y_range - 1

        for epoch in range(max_epochs):
            total_error = 0
            for data in training_data:
                # Calculate Net Input
                net = self.weights[0] * self.bias + self.weights[1] * data['x'] + self.weights[2] * data['y']
                
                # Activation (Sigmoid)
                try:
                    output = 1 / (1 + math.exp(-net))
                except OverflowError:
                    output = 0 if net < 0 else 1

                # Error
                error = data['target'] - output
                total_error += error ** 2
                
                # Derivative of Sigmoid: f'(x) = f(x) * (1 - f(x))
                derivative = output * (1 - output)
                
                # Update weights (Delta Rule)
                # w_new = w_old + learning_rate * error * derivative * input
                change_factor = learning_rate * error * derivative
                
                self.weights[0] += change_factor * self.bias
                self.weights[1] += change_factor * data['x']
                self.weights[2] += change_factor * data['y']
            
            self.errors.append(total_error)
            self.cycle_count += 1
            self.update_weights_display()
            self.draw_decision_boundary()
            self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
            self.root.update()
            # time.sleep(0.010)
            
            if total_error < min_error: # Convergence threshold
                print(f"Converged in {epoch+1} epochs. Total Error: {total_error}")
                break

        # Collect results for regression graph
        self.training_results = []
        for data in training_data:
            net = self.weights[0] * self.bias + self.weights[1] * data['x'] + self.weights[2] * data['y']
            try:
                output = 1 / (1 + math.exp(-net))
            except OverflowError:
                output = 0 if net < 0 else 1
            self.training_results.append({'target': data['target'], 'output': output})

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

    def show_regression_graph(self):
        if not self.training_results:
            messagebox.showinfo("Info", "No training results to display.")
            return
            
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Regression Graph (Target vs Output)")
        graph_window.geometry("600x400")
        
        canvas = tk.Canvas(graph_window, bg="white", width=550, height=350)
        canvas.pack(padx=20, pady=20)
        
        # Draw axes
        # Center (0,0) might be in the middle if targets are -1 to 1
        # Or bottom left if 0 to 1.
        
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
        # Find where 0 is
        zero_x = val_to_screen(0, True)
        zero_y = val_to_screen(0, False)
        
        # If 0 is within view, draw axis lines there, else draw at edges
        axis_x = zero_x if 50 <= zero_x <= 500 else 50
        axis_y = zero_y if 50 <= zero_y <= 300 else 300
        
        # X Axis line
        canvas.create_line(50, axis_y, 500, axis_y, width=1, fill="gray")
        # Y Axis line
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

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
