import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import math
import random
import csv

# --- Matrix Math Helpers ---
def mat_zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def mat_rand(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

def mat_dot(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B: raise ValueError(f"Shape mismatch: {rows_A}x{cols_A} vs {rows_B}x{cols_B}")
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

class MnistGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Handwriting Recognition")
        self.root.geometry("1000x700")
        
        self.train_data = [] # List of (label, [784 pixels])
        self.test_data = []
        self.nn = None
        self.error_history = []
        
        # Main Layout
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Side: Drawing Canvas & Visualization
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_frame, text="Draw Digit (28x28)", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.pixel_size = 15
        self.grid_size = 28
        canvas_size = self.pixel_size * self.grid_size
        
        self.canvas = tk.Canvas(left_frame, width=canvas_size, height=canvas_size, bg="black", relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        self.grid_data = [[0.0] * self.grid_size for _ in range(self.grid_size)]
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Clear Drawing", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Predict Drawing", command=self.predict_drawing).pack(side=tk.LEFT, padx=5)
        
        self.prediction_label = tk.Label(left_frame, text="Prediction: None", font=("Arial", 14, "bold"), fg="blue")
        self.prediction_label.pack(pady=10)

        # Right Side: Controls
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, anchor="n")
        
        tk.Label(controls_frame, text="Data Loading", font=("Arial", 12, "bold")).pack(pady=5)
        
        tk.Button(controls_frame, text="Load Training CSV (Select 100/digit)", command=self.load_train_csv, bg="#dddddd").pack(fill=tk.X, pady=5)
        self.train_status_label = tk.Label(controls_frame, text="Train Data: 0 samples", fg="red")
        self.train_status_label.pack(pady=2)

        tk.Button(controls_frame, text="Load Test CSV (Select 10/digit)", command=self.load_test_csv, bg="#dddddd").pack(fill=tk.X, pady=5)
        self.test_status_label = tk.Label(controls_frame, text="Test Data: 0 samples", fg="red")
        self.test_status_label.pack(pady=2)
        
        tk.Label(controls_frame, text="Configuration", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        
        tk.Label(controls_frame, text="Hidden Layers (e.g. 32):").pack(anchor="w")
        self.hidden_layers_var = tk.StringVar(value="32")
        tk.Entry(controls_frame, textvariable=self.hidden_layers_var).pack(fill=tk.X, pady=5)
        
        tk.Label(controls_frame, text="Learning Rate:").pack(anchor="w")
        self.lr_var = tk.DoubleVar(value=0.1)
        tk.Entry(controls_frame, textvariable=self.lr_var).pack(fill=tk.X, pady=5)
        
        tk.Label(controls_frame, text="Epochs:").pack(anchor="w")
        self.epochs_var = tk.IntVar(value=50)
        tk.Entry(controls_frame, textvariable=self.epochs_var).pack(fill=tk.X, pady=5)
        
        tk.Button(controls_frame, text="Initialize Network", command=self.init_network).pack(fill=tk.X, pady=5)
        tk.Button(controls_frame, text="Train Network", command=self.train, bg="#aaffaa").pack(fill=tk.X, pady=5)
        tk.Button(controls_frame, text="Test Network", command=self.test_network, bg="#aaaaff").pack(fill=tk.X, pady=5)
        tk.Button(controls_frame, text="Show Error Graph", command=self.show_error_graph, bg="#ffaaaa").pack(fill=tk.X, pady=5)
        
        tk.Label(controls_frame, text="Results", font=("Arial", 12, "bold")).pack(pady=(15, 5))
        self.accuracy_label = tk.Label(controls_frame, text="Test Accuracy: N/A")
        self.accuracy_label.pack(anchor="w")
        self.final_error_label = tk.Label(controls_frame, text="Final Error: N/A")
        self.final_error_label.pack(anchor="w")

    def paint(self, event):
        x, y = event.x, event.y
        col = x // self.pixel_size
        row = y // self.pixel_size
        
        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            # Draw on canvas
            x1 = col * self.pixel_size
            y1 = row * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="white")
            
            self.grid_data[row][col] = 1.0
            
            # Simple brush spread
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid_data[nr][nc] < 0.5:
                            self.grid_data[nr][nc] = max(self.grid_data[nr][nc], 0.5)
                            nx1 = nc * self.pixel_size
                            ny1 = nr * self.pixel_size
                            nx2 = nx1 + self.pixel_size
                            ny2 = ny1 + self.pixel_size
                            gray_val = int(self.grid_data[nr][nc] * 255)
                            color = f"#{gray_val:02x}{gray_val:02x}{gray_val:02x}"
                            self.canvas.create_rectangle(nx1, ny1, nx2, ny2, fill=color, outline=color)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid_data = [[0.0] * self.grid_size for _ in range(self.grid_size)]
        self.prediction_label.config(text="Prediction: None")

    def load_csv_data(self, samples_per_digit):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return None
            
        data = []
        counts = {i: 0 for i in range(10)}
        total_needed = samples_per_digit * 10
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                
                # Check if header is actually data
                if header:
                    try:
                        int(header[0])
                        # It's data, process it
                        label = int(header[0])
                        if counts[label] < samples_per_digit:
                            pixels = [float(p) / 255.0 for p in header[1:]]
                            data.append((label, pixels))
                            counts[label] += 1
                    except ValueError:
                        pass # It was a header
                
                for row in reader:
                    if not row: continue
                    try:
                        label = int(row[0])
                        if counts[label] < samples_per_digit:
                            pixels = [float(p) / 255.0 for p in row[1:]]
                            if len(pixels) == 784:
                                data.append((label, pixels))
                                counts[label] += 1
                    except ValueError:
                        continue
                        
                    # Check if we have enough of all
                    if all(c >= samples_per_digit for c in counts.values()):
                        break
                        
            return data, counts
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            return None, None

    def load_train_csv(self):
        data, counts = self.load_csv_data(samples_per_digit=100)
        if data:
            self.train_data = data
            self.train_status_label.config(text=f"Train Data: {len(data)} samples", fg="green")
            
            count_str = "\n".join([f"Digit {k}: {v}" for k, v in sorted(counts.items())])
            messagebox.showinfo("Training Data Loaded", f"Total: {len(data)}\n\nDistribution:\n{count_str}")

    def load_test_csv(self):
        data, counts = self.load_csv_data(samples_per_digit=10)
        if data:
            self.test_data = data
            self.test_status_label.config(text=f"Test Data: {len(data)} samples", fg="green")
            
            count_str = "\n".join([f"Digit {k}: {v}" for k, v in sorted(counts.items())])
            messagebox.showinfo("Test Data Loaded", f"Total: {len(data)}\n\nDistribution:\n{count_str}")

    def init_network(self):
        try:
            hidden_layers_str = self.hidden_layers_var.get()
            hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
            
            input_size = 784
            output_size = 10
            
            self.nn = MultiLayerPerceptron(input_size, hidden_layers, output_size)
            print("Weights initialized.")
            messagebox.showinfo("Info", f"Network Initialized.\nInput: 784\nHidden: {hidden_layers}\nOutput: 10")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid hidden layers configuration.")

    def train(self):
        if not self.train_data:
            messagebox.showwarning("Warning", "No training data loaded.")
            return
            
        if self.nn is None:
            self.init_network()
            
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        
        self.error_history = []
        
        # Prepare data
        X = [d[1] for d in self.train_data]
        y = []
        for d in self.train_data:
            label = d[0]
            vec = [0.0] * 10
            vec[label] = 1.0
            y.append(vec)
            
        print("Starting training...")
        self.root.config(cursor="watch")
        self.root.update()
        
        for epoch in range(epochs):
            # Shuffle
            combined = list(zip(X, y))
            random.shuffle(combined)
            X_shuffled, y_shuffled = zip(*combined)
            
            total_error = 0
            
            # SGD
            for i in range(len(X_shuffled)):
                sample_x = [X_shuffled[i]]
                sample_y = [y_shuffled[i]]
                
                self.nn.forward(sample_x)
                mse = self.nn.backward(sample_y, lr, momentum=0.5)
                total_error += mse
                
                if i % 50 == 0:
                    self.root.update()
            
            avg_error = total_error / len(X_shuffled)
            self.error_history.append(avg_error)
            print(f"Epoch {epoch+1}/{epochs}, Error: {avg_error:.4f}")
            self.final_error_label.config(text=f"Final Error: {avg_error:.4f}")
            
        self.root.config(cursor="")
        messagebox.showinfo("Done", "Training Finished.")

    def test_network(self):
        if self.nn is None:
            messagebox.showwarning("Warning", "Network not initialized.")
            return
            
        if not self.test_data:
            messagebox.showwarning("Warning", "No test data loaded.")
            return
            
        correct = 0
        total = len(self.test_data)
        
        for label, pixels in self.test_data:
            output = self.nn.forward([pixels])
            probs = output[0]
            prediction = probs.index(max(probs))
            
            if prediction == label:
                correct += 1
                
        accuracy = (correct / total) * 100
        self.accuracy_label.config(text=f"Test Accuracy: {accuracy:.2f}%")
        messagebox.showinfo("Test Results", f"Accuracy: {accuracy:.2f}%\nCorrect: {correct}/{total}")

    def predict_drawing(self):
        if self.nn is None:
            messagebox.showwarning("Warning", "Network not initialized/trained.")
            return
            
        # Flatten grid
        pixels = []
        for row in self.grid_data:
            for val in row:
                pixels.append(val)
        
        # Forward
        output = self.nn.forward([pixels])
        probs = output[0]
        prediction = probs.index(max(probs))
        confidence = max(probs)
        
        self.prediction_label.config(text=f"Prediction: {prediction} ({confidence:.2f})")

    def show_error_graph(self):
        if not self.error_history:
            messagebox.showinfo("Info", "No training history available.")
            return
            
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Error Graph")
        graph_window.geometry("600x400")
        
        canvas = tk.Canvas(graph_window, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Wait for window to be drawn to get actual size, or just use fixed size for simplicity
        w = 600
        h = 400
        padding = 50
        
        # Draw axes
        canvas.create_line(padding, h-padding, w-padding, h-padding, arrow=tk.LAST, width=2) # X axis
        canvas.create_line(padding, h-padding, padding, padding, arrow=tk.LAST, width=2) # Y axis
        
        # Labels
        canvas.create_text(w/2, h-20, text="Epochs", font=("Arial", 10))
        canvas.create_text(20, h/2, text="Error (MSE)", angle=90, font=("Arial", 10))
        
        max_error = max(self.error_history)
        min_error = min(self.error_history)
        if max_error == 0: max_error = 1.0
        
        # Y-axis labels (Max and Min)
        canvas.create_text(padding-25, padding, text=f"{max_error:.4f}", font=("Arial", 8))
        canvas.create_text(padding-25, h-padding, text="0.0", font=("Arial", 8))
        
        num_epochs = len(self.error_history)
        
        if num_epochs > 1:
            prev_x = padding
            prev_y = h - padding - (self.error_history[0] / max_error * (h - 2*padding))
            
            for i in range(1, num_epochs):
                x = padding + (i / (num_epochs - 1)) * (w - 2*padding)
                y = h - padding - (self.error_history[i] / max_error * (h - 2*padding))
                
                canvas.create_line(prev_x, prev_y, x, y, fill="blue", width=2)
                prev_x, prev_y = x, y
        else:
            # Single point
            y = h - padding - (self.error_history[0] / max_error * (h - 2*padding))
            canvas.create_oval(padding-2, y-2, padding+2, y+2, fill="blue")

if __name__ == "__main__":
    root = tk.Tk()
    app = MnistGUI(root)
    root.mainloop()
