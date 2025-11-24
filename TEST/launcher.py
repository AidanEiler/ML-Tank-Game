"""
launcher.py - tank combat GUI interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import driver

# --- HELPER CLASS: SCROLLABLE WINDOW ---
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, bg='#2b2b2b', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_window = ttk.Frame(self.canvas, style='TFrame')

        self.scrollable_window.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_window, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        
    def _on_mousewheel(self, event):
        # Only scroll if the content is actually taller than the window
        if self.canvas.yview() == (0.0, 1.0): return 
        
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")

    def toggle_scrolling(self, enable):
        if enable:
            self.scrollbar.pack(side="right", fill="y")
            self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        else:
            self.scrollbar.pack_forget()
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

class DarkTankLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Tank Combat AI - Command Center")
        self.root.geometry("600x600") 
        self.root.configure(bg='#2b2b2b') 

        # ==============================
        # 🎨 DARK MODE STYLING
        # ==============================
        self.style = ttk.Style()
        self.style.theme_use('clam') 
        
        self.root.unbind_class("TCombobox", "<MouseWheel>")

        bg_dark = '#2b2b2b'
        bg_darker = '#1e1e1e'
        fg_light = '#e0e0e0'
        accent_blue = '#4a90e2'
        accent_hover = '#357abd'
        
        self.hint_yellow = '#DAA520' 
        self.warn_red = '#ff6b6b'

        self.style.configure('.', background=bg_dark, foreground=fg_light)
        self.style.configure('TLabel', background=bg_dark, foreground=fg_light, font=('Helvetica', 10))
        self.style.configure('TFrame', background=bg_dark)
        self.style.configure('TLabelframe', background=bg_dark, bordercolor=accent_blue)
        self.style.configure('TLabelframe.Label', background=bg_dark, foreground=accent_blue, font=('Helvetica', 11, 'bold'))
        self.style.configure('TEntry', fieldbackground=bg_darker, foreground=fg_light, insertcolor=fg_light)
        
        self.style.configure('TCombobox', fieldbackground=bg_darker, foreground=fg_light, arrowcolor=accent_blue)
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', bg_darker)], 
                       selectbackground=[('readonly', bg_darker)], 
                       selectforeground=[('readonly', fg_light)],
                       foreground=[('readonly', fg_light)])

        self.style.configure('TCheckbutton', background=bg_dark, foreground=fg_light, 
                             indicatorcolor=bg_darker, indicatoron=accent_blue)
        self.style.map('TCheckbutton', 
                       background=[('active', bg_dark)],       
                       foreground=[('active', fg_light)],      
                       indicatorcolor=[('selected', accent_blue), ('pressed', accent_blue)])

        self.style.configure('Accent.TButton', background=accent_blue, foreground='white', font=('Helvetica', 12, 'bold'), borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', accent_hover)])

        self.desc_font = ('Helvetica', 9, 'italic')
        self.pad = {'padx': 15, 'pady': 5}
        
        self.create_gui()

    def clear_focus(self, event=None):
        self.root.focus()

    def add_desc(self, parent, text, row, col, color=None):
        if color is None: color = self.hint_yellow
        lbl = ttk.Label(parent, text=text, font=self.desc_font, foreground=color, wraplength=500, justify="left")
        lbl.grid(row=row, column=col, columnspan=2, sticky="w", padx=(25, 10), pady=(0, 15))

    def create_gui(self):
        # --- 1. BOTTOM BUTTON ---
        self.btn_frame = ttk.Frame(self.root)
        self.btn_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        
        self.start_btn = ttk.Button(self.btn_frame, text="LAUNCH SYSTEM", style="Accent.TButton", command=self.on_start)
        self.start_btn.pack(ipady=15, fill="x")

        # --- 2. SCROLLABLE AREA ---
        self.scroll_container = ScrollableFrame(self.root)
        self.scroll_container.pack(fill="both", expand=True, padx=10, pady=5)
        main_content = self.scroll_container.scrollable_window

        # ==============================
        # SECTION 1: CORE SETTINGS
        # ==============================
        frame_core = ttk.LabelFrame(main_content, text="Core Settings")
        frame_core.pack(fill="x", pady=10, padx=5)
        
        ttk.Label(frame_core, text="Operation Mode:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky="w", **self.pad)
        self.mode_var = tk.StringVar(value="Play")
        mode_cb = ttk.Combobox(frame_core, textvariable=self.mode_var, values=["Play", "Train", "Watch", "Record"], state="readonly")
        mode_cb.grid(row=0, column=1, sticky="ew", **self.pad)
        mode_cb.bind("<<ComboboxSelected>>", lambda e: [self.update_ui_layout(e), self.clear_focus()])
        self.add_desc(frame_core, "Play: Human Control.\nTrain: AI Learns.\nWatch: View AI.\nRecord: Save .mp4.", 1, 0)

        ttk.Label(frame_core, text="Processing Device:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky="w", **self.pad)
        self.device_var = tk.StringVar(value="auto")
        device_cb = ttk.Combobox(frame_core, textvariable=self.device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly")
        device_cb.grid(row=2, column=1, sticky="ew", **self.pad)
        device_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        self.add_desc(frame_core, "Auto: Best guess.\nCPU: Slow (Safe).\nCuda: Nvidia GPU.\nMPS: Mac.", 3, 0)
        
        frame_core.columnconfigure(1, weight=1)

        # ==============================
        # SECTION 2: THE ARENA
        # ==============================
        self.frame_env = ttk.LabelFrame(main_content, text="Arena Configuration")
        self.frame_env.pack(fill="x", pady=10, padx=5)
        
        ttk.Label(self.frame_env, text="Opponent Bot:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky="w", **self.pad)
        self.diff_var = tk.IntVar(value=3)
        diff_cb = ttk.Combobox(self.frame_env, textvariable=self.diff_var, values=[0, 1, 2, 3], state="readonly")
        diff_cb.grid(row=0, column=1, sticky="ew", **self.pad)
        diff_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        self.add_desc(self.frame_env, "0: Static.\n1: Zombie (Drives straight).\n2: Grunt (Avoids walls).\n3: Pro (Strafes and leads shots).", 1, 0)

        ttk.Label(self.frame_env, text="Arena Map:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky="w", **self.pad)
        self.map_style_var = tk.StringVar(value="Classic")
        map_cb = ttk.Combobox(self.frame_env, textvariable=self.map_style_var, values=["Classic", "Empty", "Dynamic", "Maze", "Forest"], state="readonly")
        map_cb.grid(row=2, column=1, sticky="ew", **self.pad)
        map_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        self.add_desc(self.frame_env, "Classic: Original.\nDynamic: Random blocks.\nMaze: City grid.\nForest: Many trees.", 3, 0)
        
        self.frame_env.columnconfigure(1, weight=1)

        # ==============================
        # SECTION 3: AI PARAMETERS
        # ==============================
        # This Container holds all optional params. We HIDE this entire container in Play mode.
        self.container_params = ttk.Frame(main_content)
        self.container_params.pack(fill="x", padx=5)

        # --- SUB-FRAME A: TRAINING INPUTS ---
        self.sub_train = ttk.LabelFrame(self.container_params, text="Training Parameters")
        
        r = 0
        ttk.Label(self.sub_train, text="Training Opponent:").grid(row=r, column=0, sticky="w", **self.pad)
        self.opponent_type_var = tk.StringVar(value="bot")
        op_cb = ttk.Combobox(self.sub_train, textvariable=self.opponent_type_var, values=["bot", "self"], state="readonly")
        op_cb.grid(row=r, column=1, sticky="ew", **self.pad); op_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        self.add_desc(self.sub_train, "Bot: Vs Script.\nSelf: AI vs AI (Advanced).", r+1, 0)
        r+=2

        ttk.Label(self.sub_train, text="Agent Level ID:").grid(row=r, column=0, sticky="w", **self.pad)
        self.level_var = tk.IntVar(value=0)
        ttk.Entry(self.sub_train, textvariable=self.level_var).grid(row=r, column=1, sticky="ew", **self.pad)
        self.add_desc(self.sub_train, "Save file number (e.g. level0).", r+1, 0)
        r+=2

        ttk.Label(self.sub_train, text="Total Timesteps:").grid(row=r, column=0, sticky="w", **self.pad)
        self.timesteps_var = tk.IntVar(value=500000)
        ttk.Entry(self.sub_train, textvariable=self.timesteps_var).grid(row=r, column=1, sticky="ew", **self.pad)
        self.add_desc(self.sub_train, "Duration. 500k ~ 1 hour.", r+1, 0)
        r+=2

        ttk.Label(self.sub_train, text="Stop Threshold:").grid(row=r, column=0, sticky="w", **self.pad)
        self.threshold_var = tk.StringVar(value="")
        ttk.Entry(self.sub_train, textvariable=self.threshold_var).grid(row=r, column=1, sticky="ew", **self.pad)
        self.add_desc(self.sub_train, "Stop if Avg Reward > X. \n\nRecommended for Bot:\n(Lvl 0: 3.3, Lvl 1: 1.25, Lvl 2: 1.5, Lvl 3: 999).", r+1, 0)
        r+=2

        ttk.Label(self.sub_train, text="Learning Rate:").grid(row=r, column=0, sticky="w", **self.pad)
        self.lr_var = tk.DoubleVar(value=0.0003)
        ttk.Entry(self.sub_train, textvariable=self.lr_var).grid(row=r, column=1, sticky="ew", **self.pad)
        self.add_desc(self.sub_train, "Brain plasticity. Default 0.0003.", r+1, 0)
        r+=2

        ttk.Label(self.sub_train, text="Training Visuals:").grid(row=r, column=0, sticky="w", **self.pad)
        self.render_mode_var = tk.StringVar(value="none")
        rend_cb = ttk.Combobox(self.sub_train, textvariable=self.render_mode_var, values=["none", "human"], state="readonly")
        rend_cb.grid(row=r, column=1, sticky="ew", **self.pad); rend_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        self.add_desc(self.sub_train, "None: Fast.\nHuman: Watch (Slow).", r+1, 0)
        
        self.sub_train.columnconfigure(1, weight=1)

        # --- SUB-FRAME B: MODEL LOADING ---
        self.sub_model = ttk.LabelFrame(self.container_params, text="Model Loading")
        
        self.lbl_model_text = tk.StringVar(value="Model Path:")
        ttk.Label(self.sub_model, textvariable=self.lbl_model_text).grid(row=0, column=0, sticky="w", **self.pad)
        
        self.model_path_var = tk.StringVar(value="auto")
        ttk.Entry(self.sub_model, textvariable=self.model_path_var).grid(row=0, column=1, sticky="ew", **self.pad)
        self.add_desc(self.sub_model, "Path to .zip file. 'auto' for fresh training.", 1, 0)
        
        self.add_desc(self.sub_model, "NOTE: Change '\\' slashes to '/'", 2, 0, color=self.warn_red)
        self.add_desc(self.sub_model, "NOTE: Remove the .zip extension", 3, 0, color=self.warn_red)
        
        self.sub_model.columnconfigure(1, weight=1)

        # --- SUB-FRAME C: RECORDING ---
        self.sub_record = ttk.LabelFrame(self.container_params, text="Recording Options")
        
        ttk.Label(self.sub_record, text="Output Filename:").grid(row=0, column=0, sticky="w", **self.pad)
        self.filename_var = tk.StringVar(value="tank_gameplay.mp4")
        ttk.Entry(self.sub_record, textvariable=self.filename_var).grid(row=0, column=1, sticky="ew", **self.pad)
        
        self.sub_record.columnconfigure(1, weight=1)

        # Initialize Layout
        self.update_ui_layout()

    def update_ui_layout(self, event=None):
        mode = self.mode_var.get()
        
        # 1. Hide sub-components
        self.sub_train.pack_forget()
        self.sub_model.pack_forget()
        self.sub_record.pack_forget()
        
        # 2. Handle Container Visibility
        if mode == "Play":
            # In Play mode, HIDE the entire container so it takes 0 space
            self.container_params.pack_forget()
        else:
            # In other modes, show the container
            self.container_params.pack(fill="x", padx=5)
            
            if mode == "Train":
                self.lbl_model_text.set("Continue From:")
                self.sub_train.pack(fill="x", pady=5)
                self.sub_model.pack(fill="x", pady=5)
            elif mode == "Watch":
                self.lbl_model_text.set("Model File:")
                self.sub_model.pack(fill="x", pady=5)
            elif mode == "Record":
                self.lbl_model_text.set("Model File:")
                self.sub_model.pack(fill="x", pady=5)
                self.sub_record.pack(fill="x", pady=5)

        # 3. Dynamic Resizing
        self.root.update_idletasks() 
        
        # Get height of the content inside the scroll frame
        content_height = self.scroll_container.scrollable_window.winfo_reqheight()
        
        # Calculate total needed height (content + button frame + some padding)
        total_needed = content_height + 80
        
        max_h = 900
        min_h = 200
        
        if total_needed > max_h:
            self.scroll_container.toggle_scrolling(True)
            final_h = max_h
        else:
            self.scroll_container.toggle_scrolling(False)
            final_h = max(min_h, total_needed)
            
        self.root.geometry(f"600x{final_h}")

    def on_start(self):
        mode = self.mode_var.get()
        thresh_str = self.threshold_var.get().strip()
        
        # Parse Stop Threshold
        try:
            # Try to convert string to float, or set to None if empty
            stop_threshold = float(thresh_str) if thresh_str else None
        except ValueError:
            # If user typed text instead of a number, show error and stop
            messagebox.showerror("Invalid Input", "Stop Threshold must be a number (e.g. 3.5) or empty.")
            return

        params = {
            "bot_difficulty": self.diff_var.get(),
            "map_style": self.map_style_var.get(),
            "device": self.device_var.get(),
            "continue_from": self.model_path_var.get(),
            "level": self.level_var.get(),
            "timesteps": self.timesteps_var.get(),
            "learning_rate": float(self.lr_var.get()),
            "n_steps": 2048,
            "batch_size": 64,
            "verbose": 1,
            "render_mode": None if self.render_mode_var.get() == "none" else "human",
            "grid_size": 600,
            "max_steps": 3000,
            "opponent_type": self.opponent_type_var.get(),
            "record_filename": self.filename_var.get(),
            "stop_threshold": stop_threshold
        }
        
        self.root.withdraw() 

        try:
            if mode == "Play":
                print(f"\n>>> Launching Human Play Mode <<<")
                driver.run_human_play(params["bot_difficulty"], params["map_style"])
            elif mode == "Train":
                print(f"\n>>> Launching AI Training on {params['device'].upper()} vs {params['opponent_type'].upper()} <<<")
                driver.run_training(params)
            elif mode == "Watch":
                print(f"\n>>> Launching Watch Mode <<<")
                driver.run_watch(params)
            elif mode == "Record":
                print(f"\n>>> Launching Record Mode <<<")
                driver.run_record(params)

        except Exception as e:
             messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.root.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = DarkTankLauncher(root)
    root.mainloop()
