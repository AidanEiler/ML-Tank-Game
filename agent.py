"""
agent.py - agent class for tank combat rl project

provides a wrapper around stable-baselines3 PPO model with:
- configurable hyperparameters
- automatic model naming based on level and wall variant
- training, saving, and loading functionality
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


class TankAgent:
    """
    wrapper class for ppo agent in tank combat environment.
    handles model creation, training, saving, and loading.
    """
    
    def __init__(
        self,
        level=0,
        walls_variant=False,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        models_dir="./models",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    ):
        """
        initialize the tank agent.
        
        args:
            level: int
                the "version" of this agent (0, 1, 2, etc.)
                used for model naming and loading
            walls_variant: bool
                whether this agent is trained with walls
                used for model naming to distinguish variants
            learning_rate: float
                ppo learning rate (default 3e-4)
            n_steps: int
                number of steps to collect before each policy update
                higher = more stable but slower updates
            batch_size: int
                mini-batch size for gradient updates
            verbose: int (0, 1, or 2)
                0 = no output
                1 = training info
                2 = debug info
            models_dir: str
                directory to save final trained models
            checkpoint_dir: str
                directory to save training checkpoints
            log_dir: str
                directory for tensorboard logs
        """
        
        self.level = level
        self.walls_variant = walls_variant
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.models_dir = models_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # model will be set during train() or load()
        self.model = None
    
    def _get_model_name(self):
        """
        generate model filename based on level and wall variant.
        
        returns:
            str like "tank_agent_level0_nowalls" or "tank_agent_level2_walls"
        """
        wall_suffix = "walls" if self.walls_variant else "nowalls"
        return f"tank_agent_level{self.level}_{wall_suffix}"
    
    def _get_model_path(self):
        """
        get full path to the model file (without .zip extension).
        """
        return os.path.join(self.models_dir, self._get_model_name())
    
    def _get_previous_model_path(self):
        """
        get path to the previous level model (for continue training).
        returns none if level is 0.
        """
        if self.level == 0:
            return None
        
        wall_suffix = "walls" if self.walls_variant else "nowalls"
        prev_name = f"tank_agent_level{self.level - 1}_{wall_suffix}"
        return os.path.join(self.models_dir, prev_name)
    
    def exists(self):
        """
        check if a saved model exists for this agent configuration.
        """
        return os.path.exists(self._get_model_path() + ".zip")
    
    def previous_exists(self):
        """
        check if the previous level model exists.
        """
        prev_path = self._get_previous_model_path()
        if prev_path is None:
            return False
        return os.path.exists(prev_path + ".zip")
    
    def train(self, env, timesteps, continue_from=None, checkpoint_freq=50000):
        """
        train the agent.
        
        args:
            env: gymnasium environment
                the tank combat environment to train in
            timesteps: int
                total number of timesteps to train for
            continue_from: str or None
                path to a model to continue training from
                if none and level > 0, will try to load previous level
                if none and level == 0, creates fresh model
            checkpoint_freq: int
                save checkpoint every n steps (default 50000)
        
        returns:
            self (for chaining)
        """
        
        # wrap environment for stable-baselines3
        monitored_env = Monitor(env, self.log_dir)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # determine what model to start from
        if continue_from is not None:
            # explicit path provided
            if not os.path.exists(continue_from + ".zip"):
                raise FileNotFoundError(f"model not found: {continue_from}.zip")
            print(f"loading model from: {continue_from}")
            self.model = PPO.load(continue_from, env=vec_env)
            
        elif self.level > 0 and self.previous_exists():
            # auto-load previous level
            prev_path = self._get_previous_model_path()
            print(f"loading previous level model: {prev_path}")
            self.model = PPO.load(prev_path, env=vec_env)
            
        elif self.level > 0 and not self.previous_exists():
            # previous level doesn't exist, warn and create fresh
            print(f"warning: level {self.level - 1} model not found")
            print("creating fresh model instead (not recommended for level > 0)")
            self.model = self._create_fresh_model(vec_env)
            
        else:
            # level 0, create fresh model
            print("creating fresh model")
            self.model = self._create_fresh_model(vec_env)
        
        # setup checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix=self._get_model_name()
        )
        
        # train
        print(f"starting training for {timesteps} timesteps...")
        print(f"  level: {self.level}")
        print(f"  walls variant: {self.walls_variant}")
        print(f"  learning rate: {self.learning_rate}")
        print(f"  n_steps: {self.n_steps}")
        print(f"  batch_size: {self.batch_size}")
        
        tb_log_name = f"PPO_level{self.level}_{'walls' if self.walls_variant else 'nowalls'}"
        
        self.model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            tb_log_name=tb_log_name
        )
        
        print("training complete")
        return self
    
    def _create_fresh_model(self, vec_env):
        """
        create a new ppo model with configured hyperparameters.
        """
        return PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            verbose=self.verbose,
            tensorboard_log=self.log_dir
        )
    
    def save(self):
        """
        save the trained model to the models directory.
        
        returns:
            str: path where model was saved
        """
        if self.model is None:
            raise RuntimeError("no model to save. call train() or load() first")
        
        save_path = self._get_model_path()
        self.model.save(save_path)
        print(f"model saved to: {save_path}.zip")
        return save_path
    
    def load(self, path=None):
        """
        load a trained model.
        
        args:
            path: str or None
                path to model file (without .zip)
                if none, loads the model for this agent's level/variant
        
        returns:
            self (for chaining)
        """
        if path is None:
            path = self._get_model_path()
        
        if not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"model not found: {path}.zip")
        
        self.model = PPO.load(path)
        print(f"model loaded from: {path}.zip")
        return self
    
    def predict(self, obs, deterministic=True):
        """
        get action prediction from the model.
        
        args:
            obs: observation from environment
            deterministic: bool
                if true, use best action (for evaluation)
                if false, sample from distribution (for exploration)
        
        returns:
            tuple: (action, state)
        """
        if self.model is None:
            raise RuntimeError("no model loaded. call train() or load() first")
        
        return self.model.predict(obs, deterministic=deterministic)


def get_agent_defaults():
    """
    return a dictionary of default agent parameters.
    useful for interactive mode to display defaults.
    """
    return {
        "level": 0,
        "walls_variant": False,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "verbose": 1,
        "timesteps": 500000,
        "continue_from": None
    }


def get_agent_options():
    """
    return a dictionary describing valid options for each parameter.
    useful for interactive mode to display choices.
    """
    return {
        "level": {
            "type": "int",
            "range": [0, 99],
            "description": "agent version number (typically matches bot difficulty trained against)"
        },
        "walls_variant": {
            "type": "bool",
            "choices": [True, False],
            "descriptions": {
                True: "model trained with walls",
                False: "model trained without walls"
            }
        },
        "learning_rate": {
            "type": "float",
            "range": [1e-6, 1e-1],
            "description": "ppo learning rate (how fast the model learns)"
        },
        "n_steps": {
            "type": "int",
            "range": [64, 8192],
            "description": "steps collected before each policy update"
        },
        "batch_size": {
            "type": "int",
            "range": [8, 512],
            "description": "mini-batch size for gradient updates"
        },
        "verbose": {
            "type": "int",
            "choices": [0, 1, 2],
            "descriptions": {
                0: "no output",
                1: "training info",
                2: "debug info"
            }
        },
        "timesteps": {
            "type": "int",
            "range": [1000, 10000000],
            "description": "total timesteps to train for"
        },
        "continue_from": {
            "type": "str",
            "description": "path to model to continue from (or 'auto' to use previous level)"
        }
    }