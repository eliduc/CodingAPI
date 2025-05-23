#!/usr/bin/env python3
"""
CodingAPI Database Browser

CodingAPI Database Browser - Version 2.1

A utility application to browse and explore the CodingAPI database.
Features:
- Browse all projects
- View project details
- Explore iterations and code for correction mode projects
- View model results for creation mode projects
- Search and filter functionality
- Export data to files
- Run code directly from the browser interface

Improvements in version 2.0:
- Added Run App button to execute code directly from the browser
- Real-time code execution with output display
- Support for viewing all iterations of code development

Improvements in version 2.1:
- Added delete functionality for projects, iterations, and model results
- Confirmation dialogs for all delete operations
- Context menus for easy deletion
- Cascade deletion support for project dependencies

Author: Claude
Date: May 2025
"""

import os
import sys
import sqlite3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import datetime
import json
import threading

class CodingAPIBrowser:
    def __init__(self, root):
        self.root = root
        self.root.title("CodingAPI Database Browser")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Database connection
        self.db_path = None
        self.conn = None
        
        # Create main UI
        self.setup_ui()
        
        # Initialize database
        self.connect_to_database()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="CodingAPI Database Browser", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Database path display and button
        db_frame = ttk.Frame(main_frame)
        db_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(db_frame, text="Database: ").pack(side=tk.LEFT)
        self.db_label = ttk.Label(db_frame, text="Not connected")
        self.db_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_btn = ttk.Button(db_frame, text="Browse...", command=self.browse_database)
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_btn = ttk.Button(db_frame, text="Refresh", command=self.refresh_data)
        self.refresh_btn.pack(side=tk.LEFT)
        
        self.delete_project_btn = ttk.Button(db_frame, text="Delete Project", 
                                           command=lambda: self.delete_selected_project())
        self.delete_project_btn.pack(side=tk.LEFT, padx=5)
        
        # Main content - split into left panel (projects) and right panel (details)
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Projects list
        left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)
        
        # Search and filter
        filter_frame = ttk.LabelFrame(left_frame, text="Filter")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Search:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(filter_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        self.search_btn = ttk.Button(filter_frame, text="Search", command=self.search_projects)
        self.search_btn.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Mode:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.mode_var = tk.StringVar(value="All")
        mode_combo = ttk.Combobox(filter_frame, textvariable=self.mode_var, 
                                   values=["All", "correction", "creation"])
        mode_combo.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self.search_projects())
        
        filter_frame.columnconfigure(1, weight=1)
        
        # Projects list
        projects_frame = ttk.LabelFrame(left_frame, text="Projects")
        projects_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for projects
        columns = ("ID", "Name", "Mode", "Created", "Status")
        self.projects_tree = ttk.Treeview(projects_frame, columns=columns, show="headings")
        
        # Define headings
        self.projects_tree.heading("ID", text="ID")
        self.projects_tree.heading("Name", text="Project Name")
        self.projects_tree.heading("Mode", text="Mode")
        self.projects_tree.heading("Created", text="Created")
        self.projects_tree.heading("Status", text="Status")
        
        # Define columns
        self.projects_tree.column("ID", width=50, anchor="center")
        self.projects_tree.column("Name", width=150)
        self.projects_tree.column("Mode", width=100, anchor="center")
        self.projects_tree.column("Created", width=150, anchor="center")
        self.projects_tree.column("Status", width=100, anchor="center")
        
        # Add scrollbar
        projects_scrollbar = ttk.Scrollbar(projects_frame, orient="vertical", command=self.projects_tree.yview)
        self.projects_tree.configure(yscrollcommand=projects_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.projects_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        projects_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.projects_tree.bind('<<TreeviewSelect>>', self.on_project_select)
        
        # Create context menu for projects
        self.projects_context_menu = tk.Menu(self.root, tearoff=0)
        self.projects_context_menu.add_command(label="Delete Project", 
                                             command=lambda: self.delete_selected_project())
        self.projects_tree.bind("<Button-3>", self.show_projects_context_menu)
        
        # Right panel - Details notebook
        self.details_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.details_frame, weight=2)
        
        # Create notebook for details
        self.notebook = ttk.Notebook(self.details_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Project details tab
        self.project_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.project_tab, text="Project Details")
        
        # Code tab
        self.code_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.code_tab, text="Code")
        
        # Create project details view
        self.setup_project_details_tab()
        
        # Create code view
        self.setup_code_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def setup_project_details_tab(self):
        """Setup the project details tab"""
        # Create a frame for the details
        details_frame = ttk.Frame(self.project_tab, padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Project info frame
        info_frame = ttk.LabelFrame(details_frame, text="Project Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create grid of labels
        fields = [
            ("Project ID:", "id_value"), 
            ("Project Name:", "name_value"),
            ("Mode:", "mode_value"),
            ("Output Directory:", "output_dir_value"),
            ("Created:", "created_value"),
            ("Completed:", "completed_value"),
            ("Coding LLM:", "coding_llm_value"),
            ("Auditing LLM:", "auditing_llm_value"),
            ("Iterations:", "iterations_value")
        ]
        
        self.details_vars = {}
        for i, (label_text, var_name) in enumerate(fields):
            row = i // 2  # Two columns
            col = (i % 2) * 2  # Skip columns for spacing
            
            ttk.Label(info_frame, text=label_text).grid(
                row=row, column=col, padx=5, pady=3, sticky="e")
            
            # Create StringVar for each field
            self.details_vars[var_name] = tk.StringVar()
            value_label = ttk.Label(info_frame, textvariable=self.details_vars[var_name])
            value_label.grid(row=row, column=col+1, padx=5, pady=3, sticky="w")
        
        # Description frame
        desc_frame = ttk.LabelFrame(details_frame, text="Program Description")
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.description_text = scrolledtext.ScrolledText(desc_frame, height=8, wrap=tk.WORD)
        self.description_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # For correction mode - iterations table
        self.iterations_frame = ttk.LabelFrame(details_frame, text="Iterations")
        
        # Create treeview for iterations
        columns = ("Iteration", "Critical", "Serious", "Non-Critical", "Suggestions", 
                   "Fixed Critical", "Fixed Serious", "Fixed Non-Critical", "Fixed Suggestions")
        self.iterations_tree = ttk.Treeview(self.iterations_frame, columns=columns, show="headings")
        
        # Define headings
        for col in columns:
            self.iterations_tree.heading(col, text=col)
            if col == "Iteration":
                self.iterations_tree.column(col, width=70, anchor="center")
            else:
                self.iterations_tree.column(col, width=80, anchor="center")
        
        # Add scrollbar
        iterations_scrollbar = ttk.Scrollbar(self.iterations_frame, orient="vertical", 
                                           command=self.iterations_tree.yview)
        self.iterations_tree.configure(yscrollcommand=iterations_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.iterations_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        iterations_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event for iterations
        self.iterations_tree.bind('<<TreeviewSelect>>', self.on_iteration_select)
        
        # Create context menu for iterations
        self.iterations_context_menu = tk.Menu(self.root, tearoff=0)
        self.iterations_context_menu.add_command(label="Delete Iteration", 
                                               command=lambda: self.delete_selected_iteration())
        self.iterations_tree.bind("<Button-3>", self.show_iterations_context_menu)
        
        # For creation mode - model results table
        self.models_frame = ttk.LabelFrame(details_frame, text="Model Results")
        
        # Create treeview for model results
        columns = ("Model", "Status")
        self.models_tree = ttk.Treeview(self.models_frame, columns=columns, show="headings")
        
        # Define headings
        self.models_tree.heading("Model", text="Model")
        self.models_tree.heading("Status", text="Status")
        
        # Define columns
        self.models_tree.column("Model", width=200)
        self.models_tree.column("Status", width=100, anchor="center")
        
        # Add scrollbar
        models_scrollbar = ttk.Scrollbar(self.models_frame, orient="vertical", 
                                        command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=models_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.models_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        models_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event for models
        self.models_tree.bind('<<TreeviewSelect>>', self.on_model_select)
        
        # Create context menu for model results
        self.models_context_menu = tk.Menu(self.root, tearoff=0)
        self.models_context_menu.add_command(label="Delete Model Result", 
                                           command=lambda: self.delete_selected_model_result())
        self.models_tree.bind("<Button-3>", self.show_models_context_menu)
            
    def setup_code_tab(self):
        """Setup the code view tab"""
        # Create a frame for the code viewer
        code_frame = ttk.Frame(self.code_tab, padding=10)
        code_frame.pack(fill=tk.BOTH, expand=True)
        
        # Code viewer with a selector
        selector_frame = ttk.Frame(code_frame)
        selector_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(selector_frame, text="View:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.view_type_var = tk.StringVar(value="Code")
        self.view_combo = ttk.Combobox(selector_frame, textvariable=self.view_type_var, 
                                        values=["Code", "Audit"])
        self.view_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.view_combo.bind("<<ComboboxSelected>>", lambda e: self.update_code_view())
        
        # Export button
        self.export_btn = ttk.Button(selector_frame, text="Export to File", command=lambda: self.export_code())
        self.export_btn.pack(side=tk.RIGHT)
        
        # Run App button
        self.run_btn = ttk.Button(selector_frame, text="Run App", command=lambda: self.run_code())
        self.run_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Code text area
        self.code_text = scrolledtext.ScrolledText(code_frame, wrap=tk.NONE, 
                                                  font=("Courier New", 10))
        self.code_text.pack(fill=tk.BOTH, expand=True)

    def run_code(self):
        """Execute the current code in the code view"""
        # Get the code from the text widget
        code = self.code_text.get(1.0, tk.END).strip()
        
        if not code:
            messagebox.showinfo("Run", "No code to execute")
            return
            
        # Check if currently viewing audit instead of code
        if self.view_type_var.get() == "Audit":
            messagebox.showinfo("Run", "Cannot run audit text. Please switch to Code view.")
            return
        
        try:
            # Create a temporary file to store the code
            import tempfile
            import os
            
            # Create temp file with .py extension
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_path = temp_file.name
                temp_file.write(code)
            
            # Create output window
            output_window = tk.Toplevel(self.root)
            output_window.title("Code Execution")
            output_window.geometry("800x600")
            output_window.minsize(600, 400)
            
            # Create output text area
            output_text = scrolledtext.ScrolledText(output_window, wrap=tk.WORD, 
                                                   bg="black", fg="white")
            output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add close button
            close_btn = ttk.Button(output_window, text="Close", 
                                   command=output_window.destroy)
            close_btn.pack(pady=10)
            
            # Display initial message
            output_text.insert(tk.END, "Running code...\n\n")
            output_window.update()
            
            # Run the code and capture output
            import subprocess
            
            # Create process
            process = subprocess.Popen(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            
            # Function to read output asynchronously
            def read_output():
                # Read stdout and stderr
                stdout, stderr = process.communicate()
                
                # Display stdout
                if stdout:
                    output_text.insert(tk.END, "=== STANDARD OUTPUT ===\n", "header")
                    output_text.insert(tk.END, stdout)
                    output_text.insert(tk.END, "\n")
                
                # Display stderr
                if stderr:
                    output_text.insert(tk.END, "=== ERRORS ===\n", "error_header")
                    output_text.insert(tk.END, stderr, "error")
                    output_text.insert(tk.END, "\n")
                
                # Display exit code
                output_text.insert(tk.END, f"\nProcess completed with exit code: {process.returncode}\n")
                
                # Configure tags
                output_text.tag_configure("header", foreground="green", font=("Courier New", 10, "bold"))
                output_text.tag_configure("error_header", foreground="red", font=("Courier New", 10, "bold"))
                output_text.tag_configure("error", foreground="red")
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # Use a separate thread to avoid freezing the UI
            threading.Thread(target=read_output, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Run Error", f"Failed to run code: {str(e)}")
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def connect_to_database(self):
        """Connect to the database and load initial data"""
        # Try to find database in default locations
        locations = [
            # Current directory
            os.path.join(os.getcwd(), "projectdb", "codingapi.db"),
            # Home directory
            os.path.expanduser("~/.codingapi/codingapi.db"),
            # Look in common output directories
            os.path.join(os.getcwd(), "Life", "projectdb", "codingapi.db"),
            os.path.join(os.getcwd(), "Life-2", "projectdb", "codingapi.db")
        ]
        
        for path in locations:
            if os.path.exists(path):
                self.db_path = path
                self.status_var.set(f"Found database at {path}")
                break
        
        if self.db_path:
            self.open_database(self.db_path)
        else:
            self.status_var.set("No database found. Please browse to select a database file.")
    
    def open_database(self, db_path):
        """Open the database and load data"""
        try:
            # Close existing connection if any
            if self.conn:
                self.conn.close()
                
            # Connect to database
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Update UI
            self.db_path = db_path
            self.db_label.config(text=db_path)
            self.status_var.set(f"Connected to database: {db_path}")
            
            # Load projects
            self.load_projects()
            
        except Exception as e:
            messagebox.showerror("Database Error", f"Error opening database: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def browse_database(self):
        """Browse for database file"""
        db_path = filedialog.askopenfilename(
            title="Select CodingAPI Database",
            filetypes=(("SQLite Database", "*.db"), ("All Files", "*.*"))
        )
        
        if db_path:
            self.open_database(db_path)
    
    def load_projects(self):
        """Load projects from the database"""
        if not self.conn:
            return
            
        # Clear existing items
        for item in self.projects_tree.get_children():
            self.projects_tree.delete(item)
            
        try:
            cursor = self.conn.cursor()
            
            # Get filter conditions
            filter_mode = self.mode_var.get()
            search_text = self.search_var.get().strip()
            
            # Build query
            query = "SELECT * FROM projects"
            params = []
            
            where_clauses = []
            
            if filter_mode != "All":
                where_clauses.append("mode = ?")
                params.append(filter_mode)
                
            if search_text:
                where_clauses.append(
                    "(project_name LIKE ? OR description LIKE ?)"
                )
                params.extend([f"%{search_text}%", f"%{search_text}%"])
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += " ORDER BY start_datetime DESC"
            
            # Execute query
            cursor.execute(query, params)
            
            # Add rows to treeview
            for row in cursor.fetchall():
                # Determine status
                status = "Completed" if row["end_datetime"] else "In Progress"
                
                # Format creation time
                created_str = "Unknown"
                if row["start_datetime"]:
                    try:
                        dt = datetime.datetime.fromisoformat(row["start_datetime"])
                        created_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        created_str = row["start_datetime"]
                
                self.projects_tree.insert("", "end", 
                    values=(row["project_id"], row["project_name"], 
                            row["mode"], created_str, status),
                    tags=("completed" if status == "Completed" else "in_progress",)
                )
                
            # Configure tag colors
            self.projects_tree.tag_configure("completed", foreground="#006400")  # Dark green
            self.projects_tree.tag_configure("in_progress", foreground="#CC7000")  # Orange
            
            # Update status
            count = len(self.projects_tree.get_children())
            self.status_var.set(f"Loaded {count} projects")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load projects: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def search_projects(self):
        """Search projects based on search criteria"""
        self.load_projects()
    
    def refresh_data(self):
        """Refresh data from database"""
        if self.db_path:
            self.open_database(self.db_path)
    
    def on_project_select(self, event):
        """Handle project selection"""
        selection = self.projects_tree.selection()
        if not selection:
            return
            
        # Get selected project ID
        project_id = self.projects_tree.item(selection[0], "values")[0]
        
        # Load project details
        self.load_project_details(project_id)
    
    def load_project_details(self, project_id):
        """Load details for selected project"""
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # Query project details
            cursor.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                return
                
            # Clear previous data
            self.clear_project_details()
            
            # Format dates
            start_date = "N/A"
            end_date = "N/A"
            
            if project["start_datetime"]:
                try:
                    dt = datetime.datetime.fromisoformat(project["start_datetime"])
                    start_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    start_date = project["start_datetime"]
                    
            if project["end_datetime"]:
                try:
                    dt = datetime.datetime.fromisoformat(project["end_datetime"])
                    end_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    end_date = project["end_datetime"]
            
            # Update details
            self.details_vars["id_value"].set(project["project_id"])
            self.details_vars["name_value"].set(project["project_name"])
            self.details_vars["mode_value"].set(project["mode"])
            self.details_vars["output_dir_value"].set(project["output_directory"])
            self.details_vars["created_value"].set(start_date)
            self.details_vars["completed_value"].set(end_date)
            
            # Mode specific fields
            if project["mode"] == "correction":
                self.details_vars["coding_llm_value"].set(project["coding_llm"] or "N/A")
                self.details_vars["auditing_llm_value"].set(project["auditing_llm"] or "N/A")
                self.details_vars["iterations_value"].set(project["iterations_count"] or "N/A")
                
                # Show iterations frame
                self.iterations_frame.pack(fill=tk.BOTH, expand=True)
                
                # Load iterations
                self.load_iterations(project_id)
                
            elif project["mode"] == "creation":
                coding_llms = project["coding_llms"] or "N/A"
                self.details_vars["coding_llm_value"].set(coding_llms)
                self.details_vars["auditing_llm_value"].set("N/A")
                self.details_vars["iterations_value"].set("N/A")
                
                # Show models frame
                self.models_frame.pack(fill=tk.BOTH, expand=True)
                
                # Load model results
                self.load_model_results(project_id)
            
            # Update description
            self.description_text.delete(1.0, tk.END)
            if project["description"]:
                self.description_text.insert(tk.END, project["description"])
            
            # Set final code in code view
            self.code_text.delete(1.0, tk.END)
            if project["final_code"]:
                self.code_text.insert(tk.END, project["final_code"])
                
            # Update status
            self.status_var.set(f"Loaded project: {project['project_name']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load project details: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def clear_project_details(self):
        """Clear project details"""
        # Clear detail fields
        for var in self.details_vars.values():
            var.set("")
            
        # Clear description
        self.description_text.delete(1.0, tk.END)
        
        # Clear code
        self.code_text.delete(1.0, tk.END)
        
        # Clear iterations
        for item in self.iterations_tree.get_children():
            self.iterations_tree.delete(item)
            
        # Clear model results
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
            
        # Hide frames
        self.iterations_frame.pack_forget()
        self.models_frame.pack_forget()
    
    def load_iterations(self, project_id):
        """Load iterations for a correction-mode project"""
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # Clear existing items
            for item in self.iterations_tree.get_children():
                self.iterations_tree.delete(item)
                
            # Query iterations
            cursor.execute(
                "SELECT * FROM iterations WHERE project_id = ? ORDER BY iteration_number",
                (project_id,)
            )
            
            # Add rows to treeview
            for row in cursor.fetchall():
                self.iterations_tree.insert("", "end", 
                    values=(
                        row["iteration_number"],
                        row["critical_count"],
                        row["serious_count"],
                        row["noncritical_count"],
                        row["suggestions_count"],
                        row["fixed_critical"],
                        row["fixed_serious"],
                        row["fixed_noncritical"],
                        row["fixed_suggestions"]
                    ),
                    tags=(str(row["iteration_id"]),)
                )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load iterations: {str(e)}")
    
    def load_model_results(self, project_id):
        """Load model results for a creation-mode project"""
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # Clear existing items
            for item in self.models_tree.get_children():
                self.models_tree.delete(item)
                
            # Query model results
            cursor.execute(
                "SELECT * FROM model_results WHERE project_id = ? ORDER BY model_name",
                (project_id,)
            )
            
            # Add rows to treeview
            for row in cursor.fetchall():
                self.models_tree.insert("", "end", 
                    values=(
                        row["model_name"],
                        row["status"]
                    ),
                    tags=(str(row["result_id"]),)
                )
                
            # Configure tag colors
            self.models_tree.tag_configure("completed", foreground="#006400")  # Dark green
            self.models_tree.tag_configure("failed", foreground="#FF0000")  # Red
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model results: {str(e)}")
    
    def on_iteration_select(self, event):
        """Handle iteration selection"""
        selection = self.iterations_tree.selection()
        if not selection:
            return
            
        # Get iteration id from tag
        iteration_id = self.iterations_tree.item(selection[0], "tags")[0]
        
        # Load iteration code and audit
        self.load_iteration_data(iteration_id)
    
    def load_iteration_data(self, iteration_id):
        """Load code and audit for a selected iteration"""
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # Query iteration data
            cursor.execute("SELECT code, audit FROM iterations WHERE iteration_id = ?", (iteration_id,))
            iteration = cursor.fetchone()
            
            if not iteration:
                return
                
            # Set code in code view based on view type
            self.update_code_view(None, iteration=iteration)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load iteration data: {str(e)}")
    
    def on_model_select(self, event):
        """Handle model selection"""
        selection = self.models_tree.selection()
        if not selection:
            return
            
        # Get result id from tag
        result_id = self.models_tree.item(selection[0], "tags")[0]
        
        # Load model code
        self.load_model_data(result_id)
    
    def load_model_data(self, result_id):
        """Load code for a selected model result"""
        if not self.conn:
            return
            
        try:
            cursor = self.conn.cursor()
            
            # Query model data
            cursor.execute("SELECT code FROM model_results WHERE result_id = ?", (result_id,))
            result = cursor.fetchone()
            
            if not result:
                return
                
            # Set code in code view
            self.code_text.delete(1.0, tk.END)
            if result["code"]:
                self.code_text.insert(tk.END, result["code"])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model data: {str(e)}")
    
    def update_code_view(self, event=None, iteration=None):
        """Update code view based on selected view type"""
        if not iteration:
            # If no iteration provided, try to get from current selection
            selection = self.iterations_tree.selection()
            if selection:
                iteration_id = self.iterations_tree.item(selection[0], "tags")[0]
                
                cursor = self.conn.cursor()
                cursor.execute("SELECT code, audit FROM iterations WHERE iteration_id = ?", (iteration_id,))
                iteration = cursor.fetchone()
            
        if not iteration:
            return
            
        view_type = self.view_type_var.get()
        
        self.code_text.delete(1.0, tk.END)
        
        if view_type == "Code" and iteration["code"]:
            self.code_text.insert(tk.END, iteration["code"])
        elif view_type == "Audit" and iteration["audit"]:
            self.code_text.insert(tk.END, iteration["audit"])
    
    def export_code(self):
        """Export current code view to a file"""
        if not self.code_text.get(1.0, tk.END).strip():
            messagebox.showinfo("Export", "No content to export")
            return
            
        # Get filename to save
        file_path = filedialog.asksaveasfilename(
            title="Export Content",
            filetypes=(
                ("Python files", "*.py"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            )
        )
        
        if not file_path:
            return
            
        try:
            # Save content to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.code_text.get(1.0, tk.END))
                
            messagebox.showinfo("Export", f"Content exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def show_projects_context_menu(self, event):
        """Show context menu for projects tree"""
        # Select the item under the cursor
        item = self.projects_tree.identify('item', event.x, event.y)
        if item:
            self.projects_tree.selection_set(item)
            self.projects_context_menu.post(event.x_root, event.y_root)

    def show_iterations_context_menu(self, event):
        """Show context menu for iterations tree"""
        # Select the item under the cursor
        item = self.iterations_tree.identify('item', event.x, event.y)
        if item:
            self.iterations_tree.selection_set(item)
            self.iterations_context_menu.post(event.x_root, event.y_root)

    def show_models_context_menu(self, event):
        """Show context menu for models tree"""
        # Select the item under the cursor
        item = self.models_tree.identify('item', event.x, event.y)
        if item:
            self.models_tree.selection_set(item)
            self.models_context_menu.post(event.x_root, event.y_root)

    def delete_selected_project(self):
        """Delete the selected project from the database"""
        selection = self.projects_tree.selection()
        if not selection:
            messagebox.showinfo("Delete Project", "Please select a project to delete.")
            return
        
        # Get project details
        project_id = self.projects_tree.item(selection[0], "values")[0]
        project_name = self.projects_tree.item(selection[0], "values")[1]
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete project '{project_name}'?\n\n"
            "This will also delete all associated iterations and model results.\n"
            "This action cannot be undone."
        )
        
        if not result:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Delete project (cascade will handle related records if foreign keys are set up)
            # Otherwise, manually delete related records first
            cursor.execute("DELETE FROM iterations WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM model_results WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            
            self.conn.commit()
            
            # Clear details view
            self.clear_project_details()
            
            # Refresh projects list
            self.load_projects()
            
            self.status_var.set(f"Deleted project: {project_name}")
            messagebox.showinfo("Delete Success", f"Project '{project_name}' has been deleted.")
            
        except Exception as e:
            self.conn.rollback()
            messagebox.showerror("Delete Error", f"Failed to delete project: {str(e)}")
            self.status_var.set(f"Error deleting project: {str(e)}")

    def delete_selected_iteration(self):
        """Delete the selected iteration from the database"""
        selection = self.iterations_tree.selection()
        if not selection:
            messagebox.showinfo("Delete Iteration", "Please select an iteration to delete.")
            return
        
        # Get iteration details
        iteration_number = self.iterations_tree.item(selection[0], "values")[0]
        iteration_id = self.iterations_tree.item(selection[0], "tags")[0]
        
        # Get current project ID
        project_selection = self.projects_tree.selection()
        if not project_selection:
            return
        project_id = self.projects_tree.item(project_selection[0], "values")[0]
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete iteration {iteration_number}?\n\n"
            "This action cannot be undone."
        )
        
        if not result:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Delete iteration
            cursor.execute("DELETE FROM iterations WHERE iteration_id = ?", (iteration_id,))
            
            # Update iterations count in project
            cursor.execute(
                "UPDATE projects SET iterations_count = iterations_count - 1 WHERE project_id = ?",
                (project_id,)
            )
            
            self.conn.commit()
            
            # Reload iterations
            self.load_iterations(project_id)
            
            # Clear code view
            self.code_text.delete(1.0, tk.END)
            
            self.status_var.set(f"Deleted iteration {iteration_number}")
            messagebox.showinfo("Delete Success", f"Iteration {iteration_number} has been deleted.")
            
        except Exception as e:
            self.conn.rollback()
            messagebox.showerror("Delete Error", f"Failed to delete iteration: {str(e)}")
            self.status_var.set(f"Error deleting iteration: {str(e)}")

    def delete_selected_model_result(self):
        """Delete the selected model result from the database"""
        selection = self.models_tree.selection()
        if not selection:
            messagebox.showinfo("Delete Model Result", "Please select a model result to delete.")
            return
        
        # Get model details
        model_name = self.models_tree.item(selection[0], "values")[0]
        result_id = self.models_tree.item(selection[0], "tags")[0]
        
        # Get current project ID
        project_selection = self.projects_tree.selection()
        if not project_selection:
            return
        project_id = self.projects_tree.item(project_selection[0], "values")[0]
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete the result for model '{model_name}'?\n\n"
            "This action cannot be undone."
        )
        
        if not result:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Delete model result
            cursor.execute("DELETE FROM model_results WHERE result_id = ?", (result_id,))
            
            self.conn.commit()
            
            # Reload model results
            self.load_model_results(project_id)
            
            # Clear code view
            self.code_text.delete(1.0, tk.END)
            
            self.status_var.set(f"Deleted model result: {model_name}")
            messagebox.showinfo("Delete Success", f"Model result for '{model_name}' has been deleted.")
            
        except Exception as e:
            self.conn.rollback()
            messagebox.showerror("Delete Error", f"Failed to delete model result: {str(e)}")
            self.status_var.set(f"Error deleting model result: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CodingAPIBrowser(root)
    root.mainloop()