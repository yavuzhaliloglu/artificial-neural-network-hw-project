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
        self.weights = [random.uniform(-1, 1) for _ in range(3)]
        self.update_weights_display()
        self.draw_decision_boundary()

    def initialize_manually(self):
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

    def update_weights_display(self):
        w0, w1, w2 = self.weights
        self.weights_label.config(text=f"{w1:.3f} (w1) x1 + {w2:.3f} (w2) x2 + {w0:.3f} (w0) = 0")

    def draw_decision_boundary(self):
        # Clear previous lines (tag 'boundary')
        self.canvas.delete("boundary")
        
        w0, w1, w2 = self.weights
        # Line eq: w1*x + w2*y + w0*bias = 0  (assuming bias input is 1)
        # y = (-w1*x - w0) / w2
        
        if w2 == 0:
            if w1 == 0: return # No line
            # x = -w0/w1 (Vertical line)
            x_val = -w0/w1
            sx1, sy1 = self.cartesian_to_screen(x_val, 10)
            sx2, sy2 = self.cartesian_to_screen(x_val, -10)
        else:
            # Calculate y for x = -10 and x = 10 (our logical bounds)
            x1 = -10
            y1 = (-w1 * x1 - w0) / w2
            x2 = 10
            y2 = (-w1 * x2 - w0) / w2
            
            sx1, sy1 = self.cartesian_to_screen(x1, y1)
            sx2, sy2 = self.cartesian_to_screen(x2, y2)
            
        self.canvas.create_line(sx1, sy1, sx2, sy2, fill="blue", width=2, tags="boundary")

    def train_binary(self):
        print("Binary training (Perceptron) selected")
        
        max_epochs = 1000
        self.cycle_count = 0
        
        # Map labels: Class 1 -> 1, Class 2 -> -1
        training_data = []
        for p in self.points:
            target = 1 if p.label == 1 else -1
            training_data.append({'x': p.x, 'y': p.y, 'target': target})
            
        if not training_data:
            messagebox.showwarning("Warning", "No data points to train!")
            return

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
                    self.weights[0] += self.learning_rate * error * self.bias
                    self.weights[1] += self.learning_rate * error * data['x']
                    self.weights[2] += self.learning_rate * error * data['y']
            
            self.cycle_count += 1
            self.update_weights_display()
            self.draw_decision_boundary()
            self.cycle_label.config(text=f"Cycles: {self.cycle_count}")
            self.root.update()
            time.sleep(0.010)
            
            if error_count == 0:
                print(f"Converged in {epoch+1} epochs.")
                break
        else:
             print("Did not converge within max epochs.")

    def train_continuous(self):
        print("Continuous training (Delta) selected")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
