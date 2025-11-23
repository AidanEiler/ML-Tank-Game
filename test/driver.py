"""
driver.py - main driver script for tank combat rl project

orchestrates environment setup and agent training.
supports two modes:
    1. cli mode: all 12 arguments provided
    2. interactive mode: prompts user for each argument

usage:
    interactive: python driver.py
    cli: python driver.py --bot-difficulty 0 --walls false --render-mode none 
                          --grid-size 600 --max-steps 3000 --level 0 
                          --learning-rate 0.0003 --n-steps 2048 --batch-size 64 
                          --verbose 1 --timesteps 500000 --continue-from auto
"""

import sys
import argparse

from environment import create_environment, get_environment_defaults, get_environment_options
from agent import TankAgent, get_agent_defaults, get_agent_options


# total number of required arguments for cli mode
REQUIRED_ARG_COUNT = 12


def parse_args():
    """
    parse command line arguments.
    returns parsed args if all 12 provided, otherwise returns none.
    """
    parser = argparse.ArgumentParser(
        description="train tank combat ai agent",
        add_help=True
    )
    
    # environment arguments
    parser.add_argument("--bot-difficulty", type=int, help="bot difficulty (0-3)")
    parser.add_argument("--walls", type=str, help="use walls (true/false)")
    parser.add_argument("--render-mode", type=str, help="render mode (none/human/rgb_array)")
    parser.add_argument("--grid-size", type=int, help="arena size in pixels")
    parser.add_argument("--max-steps", type=int, help="max steps per episode")
    
    # agent arguments
    parser.add_argument("--level", type=int, help="agent level")
    parser.add_argument("--learning-rate", type=float, help="ppo learning rate")
    parser.add_argument("--n-steps", type=int, help="steps before policy update")
    parser.add_argument("--batch-size", type=int, help="mini-batch size")
    parser.add_argument("--verbose", type=int, help="verbosity (0/1/2)")
    parser.add_argument("--timesteps", type=int, help="total training timesteps")
    parser.add_argument("--continue-from", type=str, help="model path or 'auto'")
    
    args = parser.parse_args()
    
    # count how many arguments were actually provided
    provided = sum(1 for v in vars(args).values() if v is not None)
    
    if provided == 0:
        # no args, use interactive mode
        return None
    elif provided == REQUIRED_ARG_COUNT:
        # all args provided, use cli mode
        return args
    else:
        # some but not all args provided
        print(f"error: expected {REQUIRED_ARG_COUNT} arguments, got {provided}")
        print("either provide all arguments or none (for interactive mode)")
        print("run with --help for usage information")
        sys.exit(1)


def prompt_with_options(prompt_text, options_info, default_value):
    """
    prompt user for input, displaying available options.
    
    args:
        prompt_text: str - the parameter name
        options_info: dict - info about valid options from get_*_options()
        default_value: the default value to show
    
    returns:
        user's input converted to appropriate type
    """
    print(f"\n{prompt_text}")
    print("-" * 40)
    
    opt_type = options_info.get("type", "str")
    
    # display choices if available
    if "choices" in options_info:
        choices = options_info["choices"]
        descriptions = options_info.get("descriptions", {})
        
        print("options:")
        for choice in choices:
            desc = descriptions.get(choice, "")
            if desc:
                print(f"  {choice}: {desc}")
            else:
                print(f"  {choice}")
    
    # display range if available
    elif "range" in options_info:
        range_min, range_max = options_info["range"]
        print(f"range: {range_min} to {range_max}")
    
    # display description if available
    if "description" in options_info:
        print(f"description: {options_info['description']}")
    
    # get input
    print(f"default: {default_value}")
    user_input = input(f"enter value (or press enter for default): ").strip()
    
    # use default if empty
    if user_input == "":
        return default_value
    
    # convert to appropriate type
    if opt_type == "int":
        try:
            value = int(user_input)
            # validate range if specified
            if "range" in options_info:
                range_min, range_max = options_info["range"]
                if value < range_min or value > range_max:
                    print(f"warning: value {value} outside range [{range_min}, {range_max}]")
            # validate choices if specified
            if "choices" in options_info and value not in options_info["choices"]:
                print(f"warning: value {value} not in valid choices {options_info['choices']}")
            return value
        except ValueError:
            print(f"invalid integer, using default: {default_value}")
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
            print(f"invalid float, using default: {default_value}")
            return default_value
            
    elif opt_type == "bool":
        lower = user_input.lower()
        if lower in ["true", "yes", "y", "1"]:
            return True
        elif lower in ["false", "no", "n", "0"]:
            return False
        else:
            print(f"invalid boolean, using default: {default_value}")
            return default_value
            
    else:
        # string type
        return user_input


def interactive_mode():
    """
    run interactive prompts to collect all parameters.
    
    returns:
        tuple: (env_params dict, agent_params dict)
    """
    print("=" * 50)
    print("tank combat ai - interactive configuration")
    print("=" * 50)
    
    env_defaults = get_environment_defaults()
    env_options = get_environment_options()
    agent_defaults = get_agent_defaults()
    agent_options = get_agent_options()
    
    # collect environment parameters
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
    # convert "none" string to actual None
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
    
    # collect agent parameters
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
    
    env_params = {
        "bot_difficulty": bot_difficulty,
        "use_walls": use_walls,
        "render_mode": render_mode,
        "grid_size": grid_size,
        "max_steps": max_steps
    }
    
    agent_params = {
        "level": level,
        "walls_variant": use_walls,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "verbose": verbose,
        "timesteps": timesteps,
        "continue_from": continue_from
    }
    
    return env_params, agent_params


def cli_mode(args):
    """
    extract parameters from parsed command line arguments.
    
    returns:
        tuple: (env_params dict, agent_params dict)
    """
    # convert walls string to bool
    use_walls = args.walls.lower() in ["true", "yes", "y", "1"]
    
    # convert render mode string to actual value
    render_mode = None if args.render_mode == "none" else args.render_mode
    
    env_params = {
        "bot_difficulty": args.bot_difficulty,
        "use_walls": use_walls,
        "render_mode": render_mode,
        "grid_size": args.grid_size,
        "max_steps": args.max_steps
    }
    
    agent_params = {
        "level": args.level,
        "walls_variant": use_walls,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "verbose": args.verbose,
        "timesteps": args.timesteps,
        "continue_from": args.continue_from
    }
    
    return env_params, agent_params


def generate_cli_command(env_params, agent_params):
    """
    generate the equivalent cli command for the given parameters.
    useful after interactive mode to show reproducible command.
    """
    walls_str = "true" if env_params["use_walls"] else "false"
    render_str = "none" if env_params["render_mode"] is None else env_params["render_mode"]
    continue_str = agent_params["continue_from"] if agent_params["continue_from"] else "auto"
    
    cmd = (
        f"python driver.py "
        f"--bot-difficulty {env_params['bot_difficulty']} "
        f"--walls {walls_str} "
        f"--render-mode {render_str} "
        f"--grid-size {env_params['grid_size']} "
        f"--max-steps {env_params['max_steps']} "
        f"--level {agent_params['level']} "
        f"--learning-rate {agent_params['learning_rate']} "
        f"--n-steps {agent_params['n_steps']} "
        f"--batch-size {agent_params['batch_size']} "
        f"--verbose {agent_params['verbose']} "
        f"--timesteps {agent_params['timesteps']} "
        f"--continue-from {continue_str}"
    )
    
    return cmd


def run_training(env_params, agent_params):
    """
    create environment and agent, then run training.
    """
    print("\n" + "=" * 50)
    print("starting training")
    print("=" * 50)
    
    # create environment
    print("\ncreating environment...")
    env = create_environment(
        bot_difficulty=env_params["bot_difficulty"],
        use_walls=env_params["use_walls"],
        render_mode=env_params["render_mode"],
        grid_size=env_params["grid_size"],
        max_steps=env_params["max_steps"]
    )
    
    # create agent
    print("creating agent...")
    agent = TankAgent(
        level=agent_params["level"],
        walls_variant=agent_params["walls_variant"],
        learning_rate=agent_params["learning_rate"],
        n_steps=agent_params["n_steps"],
        batch_size=agent_params["batch_size"],
        verbose=agent_params["verbose"]
    )
    
    # determine continue_from path
    continue_from = agent_params["continue_from"]
    if continue_from == "auto":
        continue_from = None  # agent.train() will auto-detect previous level
    elif continue_from == "none" or continue_from == "":
        continue_from = None
    
    # train
    try:
        agent.train(
            env=env,
            timesteps=agent_params["timesteps"],
            continue_from=continue_from
        )
        
        # save the trained model
        save_path = agent.save()
        
        print("\n" + "=" * 50)
        print("training complete!")
        print(f"model saved to: {save_path}.zip")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\ntraining interrupted by user")
        response = input("save current model? (y/n): ").strip().lower()
        if response in ["y", "yes"]:
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
    # try to parse command line arguments
    args = parse_args()
    
    if args is None:
        # no args provided, use interactive mode
        env_params, agent_params = interactive_mode()
        
        # show the equivalent cli command
        print("\n" + "=" * 50)
        print("to reproduce this configuration, use:")
        print("=" * 50)
        print(generate_cli_command(env_params, agent_params))
        print("=" * 50)
        
        # confirm before starting
        print()
        confirm = input("start training with these settings? (y/n): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("training cancelled")
            return
    else:
        # all args provided, use cli mode
        env_params, agent_params = cli_mode(args)
    
    # run training
    run_training(env_params, agent_params)


if __name__ == "__main__":
    main()