"""
launcher.py - tank combat GUI interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import driver

class DarkTankLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Tank Combat AI - Command Center")
        self.root.configure(bg='#2b2b2b') 

        # ==============================
        # 🎨 DARK MODE STYLING
        # ==============================
        self.style = ttk.Style()
        self.style.theme_use('clam') 

        bg_dark = '#2b2b2b'
        bg_darker = '#1e1e1e'
        fg_light = '#e0e0e0'
        accent_blue = '#4a90e2'
        accent_hover = '#357abd'
        
        # Colors for Text
        hint_yellow = '#DAA520' 
        hint_red = '#ff6b6b'

        self.style.configure('.', background=bg_dark, foreground=fg_light)
        self.style.configure('TLabel', background=bg_dark, foreground=fg_light, font=('Helvetica', 10))
        self.style.configure('TFrame', background=bg_dark)
        self.style.configure('TLabelframe', background=bg_dark, bordercolor=accent_blue)
        self.style.configure('TLabelframe.Label', background=bg_dark, foreground=accent_blue, font=('Helvetica', 11, 'bold'))
        self.style.configure('TEntry', fieldbackground=bg_darker, foreground=fg_light, insertcolor=fg_light)
        
        # --- COMBOBOX STYLE ---
        self.style.configure('TCombobox', fieldbackground=bg_darker, foreground=fg_light, arrowcolor=accent_blue)
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', bg_darker)], 
                       selectbackground=[('readonly', bg_darker)], # Invisible highlight
                       selectforeground=[('readonly', fg_light)],
                       foreground=[('readonly', fg_light)])

        # --- CHECKBUTTON STYLE ---
        self.style.configure('TCheckbutton', background=bg_dark, foreground=fg_light, 
                             indicatorcolor=bg_darker, indicatoron=accent_blue)
        self.style.map('TCheckbutton', 
                       background=[('active', bg_dark)],       
                       foreground=[('active', fg_light)],      
                       indicatorcolor=[('selected', accent_blue), ('pressed', accent_blue)])

        # --- BUTTON STYLE ---
        self.style.configure('Accent.TButton', background=accent_blue, foreground='white', font=('Helvetica', 12, 'bold'), borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', accent_hover)])

        # Fonts & Colors
        self.desc_font = ('Helvetica', 9, 'italic')
        self.desc_color = hint_yellow
        self.warn_color = hint_red
        self.pad = {'padx': 15, 'pady': 5}
        
        self.create_gui()

    def clear_focus(self, event=None):
        """Removes focus from widgets to hide dotted lines"""
        self.root.focus()

    def create_description(self, parent, text, row, col, colspan=2, color=None):
        if color is None: color = self.desc_color
        lbl = ttk.Label(parent, text=text, font=self.desc_font, foreground=color, wraplength=500)
        lbl.grid(row=row, column=col, columnspan=colspan, sticky="w", padx=(25, 10), pady=(0, 15))

    def create_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ==============================
        # SECTION 1: CORE SETTINGS
        # ==============================
        frame_core = ttk.LabelFrame(main_frame, text="Core Settings")
        frame_core.pack(fill="x", pady=10)
        
        ttk.Label(frame_core, text="Operation Mode:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky="w", **self.pad)
        self.mode_var = tk.StringVar(value="Play")
        mode_cb = ttk.Combobox(frame_core, textvariable=self.mode_var, values=["Play", "Train", "Watch", "Record"], state="readonly")
        mode_cb.grid(row=0, column=1, sticky="ew", **self.pad)
        mode_cb.bind("<<ComboboxSelected>>", lambda e: [self.update_ui_layout(e), self.clear_focus()])
        
        self.create_description(frame_core, 
            "Play: You control the Blue Tank vs AI. \nTrain: The AI plays against itself to learn (requires setup below).\nWatch: View saved AI. \nRecord: Save AI gameplay to .mp4.", 
            row=1, col=0)

        ttk.Label(frame_core, text="Processing Device:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky="w", **self.pad)
        self.device_var = tk.StringVar(value="auto")
        device_cb = ttk.Combobox(frame_core, textvariable=self.device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly")
        device_cb.grid(row=2, column=1, sticky="ew", **self.pad)
        device_cb.bind("<<ComboboxSelected>>", self.clear_focus)
        
        self.create_description(frame_core, 
            "Auto: Best guess. \nCPU: Slow, works everywhere (use for servers). \nCuda: Nvidia GPU (Fastest). \nMPS: Mac M1/M2 Chips.", 
            row=3, col=0)
        
        frame_core.columnconfigure(1, weight=1)

        # ==============================
        # SECTION 2: THE ARENA
        # ==============================
        self.frame_env = ttk.LabelFrame(main_frame, text="Arena Configuration")
        self.frame_env.pack(fill="x", pady=10)
        
        ttk.Label(self.frame_env, text="Opponent Bot:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky="w", **self.pad)
        self.diff_var = tk.IntVar(value=3)
        diff_cb = ttk.Combobox(self.frame_env, textvariable=self.diff_var, values=[0, 1, 2, 3], state="readonly")
        diff_cb.grid(row=0, column=1, sticky="ew", **self.pad)
        diff_cb.bind("<<ComboboxSelected>>", self.clear_focus)

        self.create_description(self.frame_env, 
            "0: Stationary Turret (Aim practice). \n1: Zombie (Drives straight at you). \n2: Grunt (Avoids walls). \n3: Pro (Strafes and leads shots).", 
            row=1, col=0)

        ttk.Label(self.frame_env, text="Arena Map:", font=('Helvetica', 10, 'bold')).grid(row=2, column=0, sticky="w", **self.pad)
        self.map_style_var = tk.StringVar(value="Classic")
        map_cb = ttk.Combobox(self.frame_env, textvariable=self.map_style_var, values=["Classic", "Empty", "Dynamic", "Maze", "Forest"], state="readonly")
        map_cb.grid(row=2, column=1, sticky="ew", **self.pad)
        map_cb.bind("<<ComboboxSelected>>", self.clear_focus)

        self.create_description(self.frame_env, 
            "Classic: Your original map (Best for old saves). \nDynamic: Random walls every round (Best for general learning). \nMaze: City grid. Forest: Many small trees.", 
            row=3, col=0)
        
        self.frame_env.columnconfigure(1, weight=1)

        # ==============================
        # SECTION 3: PARAMETERS (Dynamic)
        # ==============================
        self.frame_params = ttk.LabelFrame(main_frame, text="AI Parameters")
        
        # -- Training Specific Widgets --
        self.train_widgets = []
        
        lbl = ttk.Label(self.frame_params, text="Training Opponent:"); lbl.grid(row=0, column=0, sticky="w", **self.pad); self.train_widgets.append(lbl)
        self.opponent_type_var = tk.StringVar(value="bot")
        op_cb = ttk.Combobox(self.frame_params, textvariable=self.opponent_type_var, values=["bot", "self"], state="readonly")
        op_cb.grid(row=0, column=1, sticky="ew", **self.pad); op_cb.bind("<<ComboboxSelected>>", self.clear_focus); self.train_widgets.append(op_cb)
        
        self.create_description(self.frame_params, 
            "Bot: Trains against the Heuristic Script (Zombie/Pro). Recommended.\nSelf: Trains against a copy of itself (Experimental).", 1, 0)
        # Note: Description widgets are dynamically managed in update_ui_layout if needed, but here simpler to create and hide.
        # Since create_description packs immediately, we handle visibility in update loop.
        # For simplicity in this structure, the description above is technically row 1.
        
        lbl = ttk.Label(self.frame_params, text="Agent Level ID:"); lbl.grid(row=2, column=0, sticky="w", **self.pad); self.train_widgets.append(lbl)
        self.level_var = tk.IntVar(value=0)
        ent = ttk.Entry(self.frame_params, textvariable=self.level_var); ent.grid(row=2, column=1, sticky="ew", **self.pad); self.train_widgets.append(ent)
        
        lbl = ttk.Label(self.frame_params, text="Total Timesteps:"); lbl.grid(row=4, column=0, sticky="w", **self.pad); self.train_widgets.append(lbl)
        self.timesteps_var = tk.IntVar(value=500000)
        ent = ttk.Entry(self.frame_params, textvariable=self.timesteps_var); ent.grid(row=4, column=1, sticky="ew", **self.pad); self.train_widgets.append(ent)
        
        lbl = ttk.Label(self.frame_params, text="Learning Rate:"); lbl.grid(row=6, column=0, sticky="w", **self.pad); self.train_widgets.append(lbl)
        self.lr_var = tk.DoubleVar(value=0.0003)
        ent = ttk.Entry(self.frame_params, textvariable=self.lr_var); ent.grid(row=6, column=1, sticky="ew", **self.pad); self.train_widgets.append(ent)
        
        lbl = ttk.Label(self.frame_params, text="Training Visuals:"); lbl.grid(row=8, column=0, sticky="w", **self.pad); self.train_widgets.append(lbl)
        self.render_mode_var = tk.StringVar(value="none")
        render_cb = ttk.Combobox(self.frame_params, textvariable=self.render_mode_var, values=["none", "human"], state="readonly")
        render_cb.grid(row=8, column=1, sticky="ew", **self.pad); render_cb.bind("<<ComboboxSelected>>", self.clear_focus); self.train_widgets.append(render_cb)

        # -- Shared Model Path Widget (Train/Watch/Record) --
        self.lbl_model = ttk.Label(self.frame_params, text="Model Path:")
        self.lbl_model.grid(row=10, column=0, sticky="w", **self.pad)
        self.model_path_var = tk.StringVar(value="auto")
        self.ent_model = ttk.Entry(self.frame_params, textvariable=self.model_path_var)
        self.ent_model.grid(row=10, column=1, sticky="ew", **self.pad)
        
        # --- WARNING LABELS (RED) ---
        self.warn_1 = ttk.Label(self.frame_params, text="NOTE: Change '\\' slashes to '/' for PATHS", font=self.desc_font, foreground=self.warn_color)
        self.warn_1.grid(row=11, column=0, columnspan=2, sticky="w", padx=(25, 10))
        self.warn_2 = ttk.Label(self.frame_params, text="NOTE: Remove the .zip extension", font=self.desc_font, foreground=self.warn_color)
        self.warn_2.grid(row=12, column=0, columnspan=2, sticky="w", padx=(25, 10), pady=(0, 15))

        # -- Record Only Widget --
        self.record_widgets = []
        lbl = ttk.Label(self.frame_params, text="Output Filename:"); lbl.grid(row=13, column=0, sticky="w", **self.pad); self.record_widgets.append(lbl)
        self.filename_var = tk.StringVar(value="tank_gameplay.mp4")
        ent = ttk.Entry(self.frame_params, textvariable=self.filename_var); ent.grid(row=13, column=1, sticky="ew", **self.pad); self.record_widgets.append(ent)

        self.frame_params.columnconfigure(1, weight=1)

        # ==============================
        # LAUNCH BUTTON
        # ==============================
        self.btn_frame = ttk.Frame(main_frame)
        self.btn_frame.pack(fill="x", pady=(20, 0))
        
        self.start_btn = ttk.Button(self.btn_frame, text="🚀 LAUNCH SYSTEM", style="Accent.TButton", command=self.on_start)
        self.start_btn.pack(ipady=15, fill="x")
        
        self.update_ui_layout()

    def update_ui_layout(self, event=None):
        mode = self.mode_var.get()
        
        # 1. Hide everything first
        self.frame_params.pack_forget()
        for w in self.train_widgets: w.grid_remove()
        for w in self.record_widgets: w.grid_remove()
        self.lbl_model.grid_remove(); self.ent_model.grid_remove()
        self.warn_1.grid_remove(); self.warn_2.grid_remove()

        # 2. Show logic based on mode
        if mode != "Play":
            self.frame_params.pack(fill="x", pady=10, before=self.btn_frame)
            
            # Always show model path for Train/Watch/Record
            self.lbl_model.grid(); self.ent_model.grid(); self.warn_1.grid(); self.warn_2.grid()
            
            if mode == "Train":
                for w in self.train_widgets: w.grid()
                # Descriptions for train mode are tricky to toggle in this simple list approach
                # Ideally, they are part of train_widgets list or static. 
                # For now, the RED warnings are visible for all non-play modes which is correct.
                
            elif mode == "Record":
                for w in self.record_widgets: w.grid()

        # 3. Auto-Size Window
        self.root.update_idletasks() 
        height = self.root.winfo_reqheight() + 20 
        self.root.geometry(f"550x{height}") 

    def on_start(self):
        mode = self.mode_var.get()
        
        # Common params
        params = {
            "bot_difficulty": self.diff_var.get(),
            "map_style": self.map_style_var.get(),
            "device": self.device_var.get(),
            "continue_from": self.model_path_var.get(),
            # Defaults for training
            "level": self.level_var.get(),
            "timesteps": self.timesteps_var.get(),
            "learning_rate": self.lr_var.get(),
            "n_steps": 2048,
            "batch_size": 64,
            "verbose": 1,
            "render_mode": None if self.render_mode_var.get() == "none" else "human",
            "grid_size": 600,
            "max_steps": 3000,
            "opponent_type": self.opponent_type_var.get(),
            "record_filename": self.filename_var.get()
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