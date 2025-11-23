"""
driver.py - main driver for tank combat rl project

handles two actions:
    - play: play manually against a bot
    - train: train an ai agent against a bot

usage:
    interactive: python driver.py
    cli (play): python driver.py --action play --bot-difficulty 0 --walls false
    cli (train): python driver.py --action train --bot-difficulty 0 --walls false 
                 --render-mode none --grid-size 600 --max-steps 3000 --level 0 
                 --learning-rate 0.0003 --n-steps 2048 --batch-size 64 --verbose 1 
                 --timesteps 500000 --continue-from auto
"""

import sys
import argparse
import numpy as np
import pygame

from environment import TankCombatEnv, create_environment, get_environment_defaults, get_environment_options
from agent import TankAgent, get_agent_defaults, get_agent_options


def run_human_play(bot_difficulty, use_walls):
    """
    let a human play against the bot using keyboard controls.
    
    controls:
        wasd: move tank (supports diagonal movement)
        left/right arrows: rotate turret
        space: fire
        escape: quit
    
    args:
        bot_difficulty: int (0-3), difficulty of opponent bot
        use_walls: bool, whether to include walls in arena
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
        use_walls=use_walls,
        render_mode="human"
    )
    
    obs, info = env.reset()
    env.render()
    
    running = True
    wins, losses = 0, 0
    
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
        
        if keys[pygame.K_a]:
            action[0] = 1
        elif keys[pygame.K_d]:
            action[0] = 2
        
        if keys[pygame.K_w]:
            action[1] = 1
        elif keys[pygame.K_s]:
            action[1] = 2
        
        if keys[pygame.K_LEFT]:
            action[2] = 2
        elif keys[pygame.K_RIGHT]:
            action[2] = 1
        
        if keys[pygame.K_SPACE]:
            action[3] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            if reward > 0:
                wins += 1
                print(f"you win! (score: {wins}-{losses})")
            else:
                losses += 1
                print(f"you lose! (score: {wins}-{losses})")
            
            obs, info = env.reset()
    
    env.close()
    print(f"\nfinal score: {wins} wins - {losses} losses")


def parse_args():
    """
    parse command line arguments.
    returns (action, args) if valid, otherwise (none, none) for interactive mode.
    """
    parser = argparse.ArgumentParser(
        description="tank combat ai - play or train",
        add_help=True
    )
    
    parser.add_argument("--action", type=str, help="action to perform (play/train)")
    
    # shared args
    parser.add_argument("--bot-difficulty", type=int, help="bot difficulty (0-3)")
    parser.add_argument("--walls", type=str, help="use walls (true/false)")
    
    # train-specific args
    parser.add_argument("--render-mode", type=str, help="render mode (none/human/rgb_array)")
    parser.add_argument("--grid-size", type=int, help="arena size in pixels")
    parser.add_argument("--max-steps", type=int, help="max steps per episode")
    parser.add_argument("--level", type=int, help="agent level")
    parser.add_argument("--learning-rate", type=float, help="ppo learning rate")
    parser.add_argument("--n-steps", type=int, help="steps before policy update")
    parser.add_argument("--batch-size", type=int, help="mini-batch size")
    parser.add_argument("--verbose", type=int, help="verbosity (0/1/2)")
    parser.add_argument("--timesteps", type=int, help="total training timesteps")
    parser.add_argument("--continue-from", type=str, help="model path or 'auto'")
    
    args = parser.parse_args()
    
    if args.action is None:
        return None, None
    
    action = args.action.lower()
    
    if action == "train":
        required = [
            args.bot_difficulty, args.walls, args.render_mode, args.grid_size,
            args.max_steps, args.level, args.learning_rate, args.n_steps,
            args.batch_size, args.verbose, args.timesteps, args.continue_from
        ]
        if None in required:
            print("error: train action requires all training arguments")
            print("required: --bot-difficulty, --walls, --render-mode, --grid-size,")
            print("          --max-steps, --level, --learning-rate, --n-steps,")
            print("          --batch-size, --verbose, --timesteps, --continue-from")
            sys.exit(1)
    elif action == "play":
        required = [args.bot_difficulty, args.walls]
        if None in required:
            print("error: play action requires: --bot-difficulty, --walls")
            sys.exit(1)
    else:
        print(f"error: unknown action '{action}'. use play or train")
        sys.exit(1)
    
    return action, args


def prompt_with_options(prompt_text, options_info, default_value):
    """
    prompt user for input, displaying available options.
    accepts first letter shortcuts (e.g., 't' for 'true').
    """
    print(f"\n{prompt_text}")
    print("-" * 40)
    
    opt_type = options_info.get("type", "str")
    
    if "choices" in options_info:
        choices = options_info["choices"]
        descriptions = options_info.get("descriptions", {})
        
        print("options:")
        for choice in choices:
            desc = descriptions.get(choice, "")
            # format as [f]irst letter style for strings/bools
            if isinstance(choice, str):
                choice_display = f"[{choice[0]}]{choice[1:]}" if len(choice) > 1 else f"[{choice}]"
            elif isinstance(choice, bool):
                choice_str = "true" if choice else "false"
                choice_display = f"[{choice_str[0]}]{choice_str[1:]}"
            else:
                choice_display = str(choice)
            
            if desc:
                print(f"  {choice_display}: {desc}")
            else:
                print(f"  {choice_display}")
    
    elif "range" in options_info:
        range_min, range_max = options_info["range"]
        print(f"range: {range_min} to {range_max}")
    
    if "description" in options_info:
        print(f"description: {options_info['description']}")
    
    # format default display
    if isinstance(default_value, bool):
        default_display = "true" if default_value else "false"
    else:
        default_display = default_value
    
    print(f"default: {default_display}")
    user_input = input("enter value (or press enter for default): ").strip().lower()
    
    if user_input == "":
        return default_value
    
    if opt_type == "int":
        try:
            value = int(user_input)
            if "range" in options_info:
                range_min, range_max = options_info["range"]
                if value < range_min or value > range_max:
                    print(f"warning: value {value} outside range [{range_min}, {range_max}]")
            if "choices" in options_info and value not in options_info["choices"]:
                print(f"warning: value {value} not in valid choices {options_info['choices']}")
            return value
        except ValueError:
            print(f"invalid integer, using default: {default_display}")
            return default_value
            
    elif opt_type == "float":
        try:
            value = float(user_input)
            if "range" in options_info:
                range_min, range_max = options_info["range"]
                if value < range_min or value > range_max:
                    print(f"warning: value {value} outside range [{range_min}, {range_max}]")
            return value
        except ValueError:
            print(f"invalid float, using default: {default_display}")
            return default_value
            
    elif opt_type == "bool":
        if user_input in ["true", "t", "yes", "y", "1"]:
            return True
        elif user_input in ["false", "f", "no", "n", "0"]:
            return False
        else:
            print(f"invalid boolean, using default: {default_display}")
            return default_value
            
    else:
        return user_input


def interactive_select_action():
    """
    prompt user to select an action.
    """
    print("=" * 50)
    print("tank combat ai")
    print("=" * 50)
    
    print("\nwhat would you like to do?")
    print("-" * 40)
    print("options:")
    print("  [p]lay: play manually against a bot")
    print("  [t]rain: train an ai agent against a bot")
    print("default: train")
    
    user_input = input("enter action (or press enter for default): ").strip().lower()
    
    if user_input == "":
        return "train"
    
    if user_input.startswith("p"):
        return "play"
    elif user_input.startswith("t"):
        return "train"
    else:
        print(f"invalid action, using default: train")
        return "train"


def interactive_play():
    """
    collect human play parameters interactively.
    """
    print("\n>>> human play configuration <<<")
    
    env_options = get_environment_options()
    
    bot_difficulty = prompt_with_options(
        "bot difficulty",
        env_options["bot_difficulty"],
        0
    )
    
    use_walls = prompt_with_options(
        "use walls",
        env_options["use_walls"],
        False
    )
    
    return {
        "bot_difficulty": bot_difficulty,
        "use_walls": use_walls
    }


def interactive_train():
    """
    collect training parameters interactively.
    """
    print("\n>>> training configuration <<<")
    
    env_defaults = get_environment_defaults()
    env_options = get_environment_options()
    agent_defaults = get_agent_defaults()
    agent_options = get_agent_options()
    
    print("\n>>> environment settings <<<")
    
    bot_difficulty = prompt_with_options(
        "bot difficulty",
        env_options["bot_difficulty"],
        env_defaults["bot_difficulty"]
    )
    
    use_walls = prompt_with_options(
        "use walls",
        env_options["use_walls"],
        env_defaults["use_walls"]
    )
    
    render_mode_input = prompt_with_options(
        "render mode",
        env_options["render_mode"],
        "none"
    )
    render_mode = None if render_mode_input == "none" else render_mode_input
    
    grid_size = prompt_with_options(
        "grid size",
        env_options["grid_size"],
        env_defaults["grid_size"]
    )
    
    max_steps = prompt_with_options(
        "max steps",
        env_options["max_steps"],
        env_defaults["max_steps"]
    )
    
    print("\n>>> agent settings <<<")
    
    level = prompt_with_options(
        "agent level",
        agent_options["level"],
        agent_defaults["level"]
    )
    
    learning_rate = prompt_with_options(
        "learning rate",
        agent_options["learning_rate"],
        agent_defaults["learning_rate"]
    )
    
    n_steps = prompt_with_options(
        "n_steps (steps before policy update)",
        agent_options["n_steps"],
        agent_defaults["n_steps"]
    )
    
    batch_size = prompt_with_options(
        "batch size",
        agent_options["batch_size"],
        agent_defaults["batch_size"]
    )
    
    verbose = prompt_with_options(
        "verbose",
        agent_options["verbose"],
        agent_defaults["verbose"]
    )
    
    timesteps = prompt_with_options(
        "timesteps (total training steps)",
        agent_options["timesteps"],
        agent_defaults["timesteps"]
    )
    
    continue_from = prompt_with_options(
        "continue from (path or 'auto')",
        agent_options["continue_from"],
        "auto"
    )
    
    return {
        "bot_difficulty": bot_difficulty,
        "use_walls": use_walls,
        "render_mode": render_mode,
        "grid_size": grid_size,
        "max_steps": max_steps,
        "level": level,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "verbose": verbose,
        "timesteps": timesteps,
        "continue_from": continue_from
    }


def generate_play_command(params):
    """generate cli command for human play."""
    walls_str = "true" if params["use_walls"] else "false"
    
    return (
        f"python driver.py --action play "
        f"--bot-difficulty {params['bot_difficulty']} "
        f"--walls {walls_str}"
    )


def generate_train_command(params):
    """generate cli command for training."""
    walls_str = "true" if params["use_walls"] else "false"
    render_str = "none" if params["render_mode"] is None else params["render_mode"]
    continue_str = params["continue_from"] if params["continue_from"] else "auto"
    
    return (
        f"python driver.py --action train "
        f"--bot-difficulty {params['bot_difficulty']} "
        f"--walls {walls_str} "
        f"--render-mode {render_str} "
        f"--grid-size {params['grid_size']} "
        f"--max-steps {params['max_steps']} "
        f"--level {params['level']} "
        f"--learning-rate {params['learning_rate']} "
        f"--n-steps {params['n_steps']} "
        f"--batch-size {params['batch_size']} "
        f"--verbose {params['verbose']} "
        f"--timesteps {params['timesteps']} "
        f"--continue-from {continue_str}"
    )


def run_training(params):
    """
    create environment and agent, then run training.
    """
    print("\n" + "=" * 50)
    print("starting training")
    print("=" * 50)
    
    print("\ncreating environment...")
    env = create_environment(
        bot_difficulty=params["bot_difficulty"],
        use_walls=params["use_walls"],
        render_mode=params["render_mode"],
        grid_size=params["grid_size"],
        max_steps=params["max_steps"]
    )
    
    print("creating agent...")
    agent = TankAgent(
        level=params["level"],
        walls_variant=params["use_walls"],
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        verbose=params["verbose"]
    )
    
    continue_from = params["continue_from"]
    if continue_from == "auto":
        continue_from = None
    elif continue_from == "none" or continue_from == "":
        continue_from = None
    
    try:
        agent.train(
            env=env,
            timesteps=params["timesteps"],
            continue_from=continue_from
        )
        
        save_path = agent.save()
        
        print("\n" + "=" * 50)
        print("training complete!")
        print(f"model saved to: {save_path}.zip")
        print("=" * 50)
        
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


def main():
    """
    main entry point.
    """
    action, args = parse_args()
    
    if action is None:
        # interactive mode
        action = interactive_select_action()
        
        if action == "play":
            params = interactive_play()
            
            print("\n" + "=" * 50)
            print("to reproduce this configuration, use:")
            print("=" * 50)
            print(generate_play_command(params))
            print("=" * 50)
            
            print()
            confirm = input("start playing? [y]/n: ").strip().lower()
            if confirm in ["n", "no"]:
                print("cancelled")
                return
            
            run_human_play(params["bot_difficulty"], params["use_walls"])
        
        elif action == "train":
            params = interactive_train()
            
            print("\n" + "=" * 50)
            print("to reproduce this configuration, use:")
            print("=" * 50)
            print(generate_train_command(params))
            print("=" * 50)
            
            print()
            confirm = input("start training? [y]/n: ").strip().lower()
            if confirm in ["n", "no"]:
                print("cancelled")
                return
            
            run_training(params)
    
    else:
        # cli mode
        if action == "play":
            use_walls = args.walls.lower() in ["true", "yes", "y", "1"]
            run_human_play(args.bot_difficulty, use_walls)
        elif action == "train":
            use_walls = args.walls.lower() in ["true", "yes", "y", "1"]
            render_mode = None if args.render_mode == "none" else args.render_mode
            
            params = {
                "bot_difficulty": args.bot_difficulty,
                "use_walls": use_walls,
                "render_mode": render_mode,
                "grid_size": args.grid_size,
                "max_steps": args.max_steps,
                "level": args.level,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "verbose": args.verbose,
                "timesteps": args.timesteps,
                "continue_from": args.continue_from
            }
            run_training(params)


if __name__ == "__main__":
    main()