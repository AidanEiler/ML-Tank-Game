"""
driver.py - main driver for tank combat rl project

Features:
- Play Mode (Human vs Bot)
- Train Mode (AI vs Bot OR AI vs Self)
- Watch Mode (Watch trained AI play)
- Record Mode (Save AI gameplay to MP4)
- Device Selection (CPU/GPU) with CRASH PROTECTION
- Full .zip loading compatibility
- Interactive Text Menu
"""

import sys
import argparse
import os
import numpy as np
import pygame
import imageio 
import torch 
from stable_baselines3 import PPO 

from environment import TankCombatEnv, create_environment, get_environment_defaults, get_environment_options
from agent import TankAgent, get_agent_defaults, get_agent_options


# =============================================================================
# HELPER: DEVICE "LIE DETECTOR"
# =============================================================================
def get_safe_device(requested_device):
    """
    1. Checks if user specifically asked for CPU.
    2. If 'auto' or 'cuda', it attempts a tiny math operation on the GPU.
    3. If that math fails (e.g. "No Kernel Image"), it forces CPU.
    """
    if requested_device == "cpu":
        return "cpu"

    # If torch thinks there is no GPU, verify CPU usage
    if not torch.cuda.is_available():
        if requested_device == "cuda":
            print("⚠️ Warning: CUDA requested but not available. Switching to CPU.")
        return "cpu"

    # --- THE VIBE CHECK ---
    # Torch thinks there is a GPU. Let's see if it actually works.
    try:
        # Try to allocate 1 tiny number on the GPU
        x = torch.tensor([1.0], device="cuda")
        # Try to do math with it
        y = x * 2
        # If we get here, the GPU is real and working.
        print(f"✅ GPU Verified: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except Exception as e:
        # This catches your "No Kernel Image" error!
        print(f"⚠️ GPU Hardware Error Detected: {e}")
        print("🔄 Your computer's GPU is incompatible. SWITCHING TO CPU.")
        return "cpu"


# =============================================================================
# 1. HUMAN PLAY
# =============================================================================
def run_human_play(bot_difficulty, map_style):
    """
    Human play doesn't use Neural Nets, so it doesn't need device checks.
    """
    print("\n" + "=" * 50)
    print("human play mode")
    print("=" * 50)
    print("\ncontrols:")
    print("  wasd: move tank")
    print("  left/right arrows: rotate turret")
    print("  space: fire")
    print("  escape: quit")
    print("=" * 50)
    
    env = TankCombatEnv(
        bot_difficulty=bot_difficulty,
        map_style=map_style, 
        render_mode="human",
        opponent_type="bot" 
    )
    
    obs, info = env.reset()
    env.render()
    
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        if not running:
            break
        
        keys = pygame.key.get_pressed()
        
        action = np.array([0, 0, 0, 0])
        
        if keys[pygame.K_a]: action[0] = 1 
        elif keys[pygame.K_d]: action[0] = 2 
        
        if keys[pygame.K_w]: action[1] = 1 
        elif keys[pygame.K_s]: action[1] = 2 
        
        if keys[pygame.K_LEFT]: action[2] = 1  
        elif keys[pygame.K_RIGHT]: action[2] = 2  
        
        if keys[pygame.K_SPACE]: action[3] = 1 
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()


# =============================================================================
# 2. TRAINING
# =============================================================================
def run_training(params):
    print("\n" + "=" * 50)
    print("starting training")
    print("=" * 50)
    
    # --- STEP 1: GET SAFE DEVICE ---
    safe_device = get_safe_device(params["device"])
    
    # --- LOGIC TO HANDLE SELF-PLAY OPPONENT LOADING ---
    opponent_policy = None
    
    if params.get("opponent_type") == "self":
        print("\n[SELF-PLAY] Attempting to load opponent model...")
        
        load_path = None
        if params["continue_from"] and params["continue_from"] != "auto":
            load_path = params["continue_from"]
        else:
            style_tag = "walls"
            if params.get("map_style") == "Empty": style_tag = "nowalls"
            
            lvl = params["level"]
            candidates = [
                f"models/tank_agent_level{lvl}_{style_tag}",
                f"models/tank_agent_level{max(0, lvl-1)}_{style_tag}"
            ]
            
            for c in candidates:
                if os.path.exists(c + ".zip"):
                    load_path = c
                    break
        
        if load_path and os.path.exists(load_path + ".zip"):
            print(f"[SELF-PLAY] Loaded opponent from: {load_path}")
            try:
                # Use SAFE DEVICE
                opponent_policy = PPO.load(load_path, device=safe_device)
            except Exception as e:
                print(f"[SELF-PLAY] Failed to load opponent: {e}")
                print("Reverting to Random Opponent.")
        else:
            print(f"[SELF-PLAY] No model found. Opponent will be Random.")

    env = TankCombatEnv(
        bot_difficulty=params["bot_difficulty"],
        map_style=params.get("map_style", "Classic"), 
        render_mode=params["render_mode"],
        grid_size=params["grid_size"],
        max_steps=params["max_steps"],
        opponent_type=params.get("opponent_type", "bot"),
        opponent_policy=opponent_policy
    )
    
    use_walls_bool = True
    if params.get("map_style") == "Empty": 
        use_walls_bool = False

    agent = TankAgent(
        level=params["level"],
        walls_variant=use_walls_bool,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        device=safe_device
    )
    
    continue_from = params["continue_from"]
    if continue_from == "auto": continue_from = None
    elif continue_from == "none" or continue_from == "": continue_from = None
    
    try:
        agent.train(
            env=env,
            timesteps=params["timesteps"],
            continue_from=continue_from
        )
        save_path = agent.save()
        print(f"model saved to: {save_path}.zip")
        
    except KeyboardInterrupt:
        print("\n\ntraining interrupted by user")
        response = input("save current model? [y]/n: ").strip().lower()
        if response not in ["n", "no"]:
            save_path = agent.save()
            print(f"model saved to: {save_path}.zip")
        else:
            print("model not saved")
            
    finally:
        env.close()


# =============================================================================
# 3. WATCH MODE
# =============================================================================
def run_watch(params):
    model_path = params["continue_from"]
    if not model_path or model_path == "auto" or not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Could not find model at {model_path}.zip")
        return

    # --- STEP 1: GET SAFE DEVICE ---
    safe_device = get_safe_device(params["device"])
    print(f"👀 Watching Model: {model_path} (Device: {safe_device})")
    
    env = TankCombatEnv(
        bot_difficulty=params["bot_difficulty"],
        map_style=params["map_style"],
        render_mode="human",
        opponent_type="bot"
    )
    
    try:
        # Use SAFE DEVICE
        model = PPO.load(model_path, device=safe_device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return
    
    for episode in range(5): 
        obs, _ = env.reset()
        env.render() # Ensure window is init
        
        terminated = False
        truncated = False
        score = 0
        print(f"Match {episode + 1} Started...")
        
        while not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            env.render()
            
        print(f"Match Finished! Score: {score:.2f}")
    
    env.close()


# =============================================================================
# 4. RECORD MODE
# =============================================================================
def run_record(params):
    model_path = params["continue_from"]
    filename = params.get("record_filename", "tank_gameplay.mp4")
    folder = "recordings"
    
    if not model_path or model_path == "auto" or not os.path.exists(model_path + ".zip"):
        print(f"❌ Error: Could not find model at {model_path}.zip")
        return

    os.makedirs(folder, exist_ok=True)
    video_path = os.path.join(folder, filename)
    
    # --- STEP 1: GET SAFE DEVICE ---
    safe_device = get_safe_device(params["device"])
    print(f"🎥 Recording to: {video_path} (Device: {safe_device})")
    
    env = TankCombatEnv(
        bot_difficulty=params["bot_difficulty"],
        map_style=params["map_style"],
        render_mode="rgb_array", 
        opponent_type="bot"
    )
    
    try:
        # Use SAFE DEVICE
        model = PPO.load(model_path, device=safe_device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return
    
    try:
        writer = imageio.get_writer(video_path, fps=60)
    except ImportError:
        print("❌ Error: 'imageio' not found. Run: pip install imageio imageio[ffmpeg]")
        return

    for episode in range(3):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        print(f"Recording Match {episode + 1}...")
        
        writer.append_data(env.render())
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            frame = env.render()
            writer.append_data(frame)
            
    writer.close()
    env.close()
    print(f"✅ Video saved to {video_path}")


# =============================================================================
# INTERACTIVE CLI TOOLS
# =============================================================================

def prompt_with_options(prompt_text, options_info, default_value):
    print(f"\n{prompt_text}")
    print("-" * 40)
    opt_type = options_info.get("type", "str")
    if "choices" in options_info:
        choices = options_info["choices"]
        descriptions = options_info.get("descriptions", {})
        print("options:")
        for choice in choices:
            desc = descriptions.get(choice, "")
            if isinstance(choice, str):
                choice_display = f"[{choice[0]}]{choice[1:]}" if len(choice) > 1 else f"[{choice}]"
            elif isinstance(choice, bool):
                choice_str = "true" if choice else "false"
                choice_display = f"[{choice_str[0]}]{choice_str[1:]}"
            else:
                choice_display = str(choice)
            if desc: print(f"  {choice_display}: {desc}")
            else: print(f"  {choice_display}")
    elif "range" in options_info:
        range_min, range_max = options_info["range"]
        print(f"range: {range_min} to {range_max}")
    if "description" in options_info:
        print(f"description: {options_info['description']}")
    if isinstance(default_value, bool):
        default_display = "true" if default_value else "false"
    else:
        default_display = default_value
    print(f"default: {default_display}")
    user_input = input("enter value (or press enter for default): ").strip().lower()
    if user_input == "": return default_value
    if opt_type == "int":
        try: return int(user_input)
        except ValueError: return default_value
    elif opt_type == "float":
        try: return float(user_input)
        except ValueError: return default_value
    elif opt_type == "bool":
        if user_input in ["true", "t", "yes", "y", "1"]: return True
        elif user_input in ["false", "f", "no", "n", "0"]: return False
        else: return default_value
    else:
        if "choices" in options_info:
             for choice in options_info["choices"]:
                 if str(choice).startswith(user_input): return choice
        return user_input

def interactive_select_action():
    print("=" * 50)
    print("tank combat ai")
    print("=" * 50)
    print("\nwhat would you like to do?")
    print("  [p]lay: play manually against a bot")
    print("  [t]rain: train an ai agent against a bot")
    print("  [w]atch: watch a trained agent play")
    print("  [r]ecord: record a trained agent to video")
    
    user_input = input("enter action: ").strip().lower()
    if user_input.startswith("p"): return "play"
    elif user_input.startswith("t"): return "train"
    elif user_input.startswith("w"): return "watch"
    elif user_input.startswith("r"): return "record"
    return "train"

def interactive_play():
    env_options = get_environment_options()
    bot_difficulty = prompt_with_options("bot difficulty", env_options["bot_difficulty"], 0)
    map_style = prompt_with_options("map style", env_options["map_style"], "Classic")
    return {"bot_difficulty": bot_difficulty, "map_style": map_style}

def interactive_train():
    print("\n>>> training configuration <<<")
    env_defaults = get_environment_defaults()
    env_options = get_environment_options()
    agent_defaults = get_agent_defaults()
    agent_options = get_agent_options()
    
    print("\n>>> environment settings <<<")
    bot_difficulty = prompt_with_options("bot difficulty", env_options["bot_difficulty"], env_defaults["bot_difficulty"])
    map_style = prompt_with_options("map style", env_options["map_style"], "Classic")
    render_mode_input = prompt_with_options("render mode", env_options["render_mode"], "none")
    render_mode = None if render_mode_input == "none" else render_mode_input
    grid_size = prompt_with_options("grid size", env_options["grid_size"], env_defaults["grid_size"])
    max_steps = prompt_with_options("max steps", env_options["max_steps"], env_defaults["max_steps"])
    
    print("\n>>> agent settings <<<")
    level = prompt_with_options("agent level", agent_options["level"], agent_defaults["level"])
    learning_rate = prompt_with_options("learning rate", agent_options["learning_rate"], agent_defaults["learning_rate"])
    n_steps = prompt_with_options("n_steps", agent_options["n_steps"], agent_defaults["n_steps"])
    batch_size = prompt_with_options("batch size", agent_options["batch_size"], agent_defaults["batch_size"])
    verbose = prompt_with_options("verbose", agent_options["verbose"], agent_defaults["verbose"])
    timesteps = prompt_with_options("timesteps", agent_options["timesteps"], agent_defaults["timesteps"])
    continue_from = prompt_with_options("continue from", agent_options["continue_from"], "auto")
    device = prompt_with_options("device (cpu/cuda)", agent_options["device"], agent_defaults["device"])
    opponent_type = prompt_with_options("opponent type (bot/self)", {"type": "str", "choices": ["bot", "self"]}, "bot")
    
    return {
        "bot_difficulty": bot_difficulty,
        "map_style": map_style,
        "render_mode": render_mode,
        "grid_size": grid_size,
        "max_steps": max_steps,
        "level": level,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "verbose": verbose,
        "timesteps": timesteps,
        "continue_from": continue_from,
        "device": device,
        "opponent_type": opponent_type
    }

def interactive_watch_or_record():
    print("\n>>> watch/record configuration <<<")
    env_options = get_environment_options()
    agent_options = get_agent_options()
    
    bot_difficulty = prompt_with_options("bot difficulty", env_options["bot_difficulty"], 0)
    map_style = prompt_with_options("map style", env_options["map_style"], "Classic")
    continue_from = prompt_with_options("model path (REQUIRED)", agent_options["continue_from"], "")
    device = prompt_with_options("device (cpu/cuda)", agent_options["device"], "auto")
    
    return {
        "bot_difficulty": bot_difficulty,
        "map_style": map_style,
        "continue_from": continue_from,
        "device": device,
        "record_filename": "tank_gameplay.mp4" 
    }

def generate_train_command(params):
    walls_str = "false" if params.get("map_style") == "Empty" else "true"
    render_str = "none" if params["render_mode"] is None else params["render_mode"]
    continue_str = params["continue_from"] if params["continue_from"] else "auto"
    
    return (
        f"python driver.py --action train "
        f"--bot-difficulty {params['bot_difficulty']} "
        f"--map-style {params['map_style']} "
        f"--render-mode {render_str} "
        f"--grid-size {params['grid_size']} "
        f"--max-steps {params['max_steps']} "
        f"--level {params['level']} "
        f"--learning-rate {params['learning_rate']} "
        f"--n-steps {params['n_steps']} "
        f"--batch-size {params['batch_size']} "
        f"--verbose {params['verbose']} "
        f"--timesteps {params['timesteps']} "
        f"--continue-from {continue_str} "
        f"--device {params['device']} "
        f"--opponent-type {params['opponent_type']}"
    )


# =============================================================================
# ARGUMENT PARSING & MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="tank combat ai")
    parser.add_argument("--action", type=str, help="action to perform (play/train/watch/record)")
    
    # Shared args
    parser.add_argument("--bot-difficulty", type=int, help="bot difficulty (0-3)")
    parser.add_argument("--walls", type=str, help="use walls (true/false) [LEGACY]") 
    parser.add_argument("--map-style", type=str, help="map style", default="Classic")
    
    # Train args
    parser.add_argument("--render-mode", type=str, help="render mode")
    parser.add_argument("--grid-size", type=int, help="arena size")
    parser.add_argument("--max-steps", type=int, help="max steps")
    parser.add_argument("--level", type=int, help="agent level")
    parser.add_argument("--learning-rate", type=float, help="lr")
    parser.add_argument("--n-steps", type=int, help="n_steps")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--verbose", type=int, help="verbose")
    parser.add_argument("--timesteps", type=int, help="timesteps")
    parser.add_argument("--continue-from", type=str, help="model path")
    parser.add_argument("--device", type=str, help="device", default="auto")
    parser.add_argument("--opponent-type", type=str, help="opponent", default="bot")
    
    # Record args
    parser.add_argument("--record-filename", type=str, help="output filename", default="tank_gameplay.mp4")
    
    args = parser.parse_args()
    if args.action is None: return None, None
    return args.action.lower(), args


def main():
    action, args = parse_args()
    
    # Interactive Mode (if no args provided)
    if action is None:
        action = interactive_select_action()
        if action == "play":
            params = interactive_play()
            run_human_play(params["bot_difficulty"], params["map_style"])
        elif action == "train":
            params = interactive_train()
            print("\n" + "=" * 50)
            print("command to reproduce:")
            print(generate_train_command(params))
            print("=" * 50)
            if input("\nstart training? [y]/n: ").lower() not in ["n", "no"]:
                run_training(params)
        elif action == "watch":
            params = interactive_watch_or_record()
            run_watch(params)
        elif action == "record":
            params = interactive_watch_or_record()
            run_record(params)
                
    # CLI Mode (if args provided)
    else:
        if action == "play":
            style = args.map_style
            if args.walls and args.walls.lower() == "false": style = "Empty"
            run_human_play(args.bot_difficulty, style)
            
        elif action == "train":
            render_mode = None if args.render_mode == "none" else args.render_mode
            style = args.map_style
            if args.walls and args.walls.lower() == "false": style = "Empty"

            params = {
                "bot_difficulty": args.bot_difficulty,
                "map_style": style,
                "render_mode": render_mode,
                "grid_size": args.grid_size,
                "max_steps": args.max_steps,
                "level": args.level,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "verbose": args.verbose,
                "timesteps": args.timesteps,
                "continue_from": args.continue_from,
                "device": args.device,
                "opponent_type": args.opponent_type
            }
            run_training(params)
            
        elif action == "watch":
            params = {
                "bot_difficulty": args.bot_difficulty,
                "map_style": args.map_style,
                "continue_from": args.continue_from,
                "device": args.device
            }
            run_watch(params)
            
        elif action == "record":
            params = {
                "bot_difficulty": args.bot_difficulty,
                "map_style": args.map_style,
                "continue_from": args.continue_from,
                "device": args.device,
                "record_filename": args.record_filename
            }
            run_record(params)

if __name__ == "__main__":
    main()