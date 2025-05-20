import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import torch
import os
import json
from collections import OrderedDict
import numpy as np

class ModelViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch Model Viewer")
        self.root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection area
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Label(file_frame, text="Model File:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Output area with tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different aspects of the model
        self.summary_tab = ttk.Frame(self.notebook)
        self.params_tab = ttk.Frame(self.notebook)
        self.structure_tab = ttk.Frame(self.notebook)
        self.meta_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.summary_tab, text="Summary")
        self.notebook.add(self.params_tab, text="Parameters")
        self.notebook.add(self.structure_tab, text="Structure")
        self.notebook.add(self.meta_tab, text="Metadata")
        
        # Create text widgets for each tab
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        self.params_text = scrolledtext.ScrolledText(self.params_tab, wrap=tk.WORD)
        self.params_text.pack(fill=tk.BOTH, expand=True)
        
        self.structure_text = scrolledtext.ScrolledText(self.structure_tab, wrap=tk.WORD)
        self.structure_text.pack(fill=tk.BOTH, expand=True)
        
        self.meta_text = scrolledtext.ScrolledText(self.meta_tab, wrap=tk.WORD)
        self.meta_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottom buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_display).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit", command=root.quit).pack(side=tk.RIGHT, padx=5)
        
        # Set monospace font for all text widgets
        font = ("Courier", 10)
        self.summary_text.configure(font=font)
        self.params_text.configure(font=font)
        self.structure_text.configure(font=font)
        self.meta_text.configure(font=font)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select PyTorch Model File",
            filetypes=[("PyTorch Files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def clear_display(self):
        self.summary_text.delete(1.0, tk.END)
        self.params_text.delete(1.0, tk.END)
        self.structure_text.delete(1.0, tk.END)
        self.meta_text.delete(1.0, tk.END)
    
    def load_model(self):
        file_path = self.file_path.get()
        if not file_path or not os.path.exists(file_path):
            tk.messagebox.showerror("Error", "Please select a valid PyTorch model file.")
            return
        
        self.clear_display()
        
        try:
            # Load the model checkpoint
            checkpoint = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
            self.analyze_checkpoint(checkpoint)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def analyze_checkpoint(self, checkpoint):
        # Summary tab
        self.summary_text.insert(tk.END, "MODEL SUMMARY\n")
        self.summary_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Get file info
        file_path = self.file_path.get()
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # In MB
        self.summary_text.insert(tk.END, f"File: {os.path.basename(file_path)}\n")
        self.summary_text.insert(tk.END, f"Size: {file_size:.2f} MB\n\n")
        
        # Check what's in the checkpoint
        self.summary_text.insert(tk.END, "Checkpoint contains:\n")
        for key in checkpoint.keys():
            self.summary_text.insert(tk.END, f"- {key}\n")
        
        # Model architecture if available
        if 'model_state_dict' in checkpoint:
            self.summary_text.insert(tk.END, "\nModel Architecture:\n")
            total_params = 0
            for name, param in checkpoint['model_state_dict'].items():
                param_size = np.prod(param.shape)
                total_params += param_size
                self.summary_text.insert(tk.END, f"- {name}: {param.shape}, {param_size:,} parameters\n")
            
            self.summary_text.insert(tk.END, f"\nTotal parameters: {total_params:,}\n")
        
        # Training stats if available
        if 'train_iterations' in checkpoint:
            self.summary_text.insert(tk.END, f"\nTraining iterations: {checkpoint['train_iterations']}\n")
        
        if 'total_games' in checkpoint:
            self.summary_text.insert(tk.END, f"Total games trained on: {checkpoint['total_games']}\n")
        
        # Parameters tab
        self.params_text.insert(tk.END, "MODEL PARAMETERS\n")
        self.params_text.insert(tk.END, "=" * 80 + "\n\n")
        
        if 'model_state_dict' in checkpoint:
            self.display_state_dict(checkpoint['model_state_dict'], self.params_text)
        
        # Structure tab
        self.structure_text.insert(tk.END, "MODEL STRUCTURE\n")
        self.structure_text.insert(tk.END, "=" * 80 + "\n\n")
        
        if 'model_state_dict' in checkpoint:
            # Group parameters by layers
            layers = {}
            for name, param in checkpoint['model_state_dict'].items():
                parts = name.split('.')
                if len(parts) > 1:
                    layer_name = parts[0]
                    if layer_name not in layers:
                        layers[layer_name] = []
                    layers[layer_name].append((name, param))
            
            # Display layers
            for layer_name, params in layers.items():
                self.structure_text.insert(tk.END, f"Layer: {layer_name}\n")
                for name, param in params:
                    shape_str = 'x'.join(str(x) for x in param.shape)
                    self.structure_text.insert(tk.END, f"  - {name.split('.', 1)[1]}: {shape_str}\n")
                self.structure_text.insert(tk.END, "\n")
        
        # Metadata tab
        self.meta_text.insert(tk.END, "MODEL METADATA\n")
        self.meta_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Display optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.meta_text.insert(tk.END, "Optimizer State:\n")
            optimizer = checkpoint['optimizer_state_dict']
            if 'param_groups' in optimizer:
                for i, group in enumerate(optimizer['param_groups']):
                    self.meta_text.insert(tk.END, f"Parameter Group {i}:\n")
                    for key, value in group.items():
                        if key != 'params':  # Skip the parameter indices
                            self.meta_text.insert(tk.END, f"  {key}: {value}\n")
            self.meta_text.insert(tk.END, "\n")
        
        # Display hyperparameters if available
        if 'hyperparams' in checkpoint:
            self.meta_text.insert(tk.END, "Hyperparameters:\n")
            for key, value in checkpoint['hyperparams'].items():
                self.meta_text.insert(tk.END, f"  {key}: {value}\n")
        
        # Display any other metadata
        self.meta_text.insert(tk.END, "\nOther Metadata:\n")
        for key, value in checkpoint.items():
            if key not in ['model_state_dict', 'optimizer_state_dict', 'hyperparams']:
                # Handle different types of values
                if isinstance(value, (int, float, str, bool)):
                    self.meta_text.insert(tk.END, f"  {key}: {value}\n")
                elif isinstance(value, dict):
                    self.meta_text.insert(tk.END, f"  {key}: {type(value).__name__} with {len(value)} items\n")
                elif isinstance(value, (list, tuple)):
                    self.meta_text.insert(tk.END, f"  {key}: {type(value).__name__} with {len(value)} elements\n")
                else:
                    self.meta_text.insert(tk.END, f"  {key}: {type(value).__name__}\n")
    
    def display_state_dict(self, state_dict, text_widget):
        for i, (name, param) in enumerate(state_dict.items()):
            text_widget.insert(tk.END, f"{i+1}. {name}:\n")
            text_widget.insert(tk.END, f"   Shape: {param.shape}\n")
            text_widget.insert(tk.END, f"   Type: {param.dtype}\n")
            
            # Display some statistics about the parameter values
            tensor = param.detach().numpy()
            text_widget.insert(tk.END, f"   Min: {tensor.min():.6f}, Max: {tensor.max():.6f}\n")
            text_widget.insert(tk.END, f"   Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}\n")
            
            # Display a small sample of values for each parameter
            if len(param.shape) > 0:
                flat = tensor.flatten()
                sample_size = min(5, len(flat))
                sample = flat[:sample_size]
                text_widget.insert(tk.END, f"   Sample values: {[float(f'{x:.4f}') for x in sample]}\n")
            
            text_widget.insert(tk.END, "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelViewerApp(root)
    root.mainloop()