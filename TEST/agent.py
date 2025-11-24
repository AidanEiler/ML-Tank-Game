"""
agent.py - agent class for tank combat rl project

provides a wrapper around stable-baselines3 PPO model with:
- configurable hyperparameters
- automatic model naming based on level and wall variant
- training, saving, and loading functionality
- device selection (cpu/gpu)
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
        device="auto",  
        models_dir="./models",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    ):
        """
        initialize the tank agent.
        """
        self.level = level
        self.walls_variant = walls_variant
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device 
        self.models_dir = models_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.model = None
    
    def _get_model_name(self):
        wall_suffix = "walls" if self.walls_variant else "nowalls"
        return f"tank_agent_level{self.level}_{wall_suffix}"
    
    def _get_model_path(self):
        return os.path.join(self.models_dir, self._get_model_name())
    
    def _get_previous_model_path(self):
        if self.level == 0:
            return None
        wall_suffix = "walls" if self.walls_variant else "nowalls"
        prev_name = f"tank_agent_level{self.level - 1}_{wall_suffix}"
        return os.path.join(self.models_dir, prev_name)
    
    def exists(self):
        return os.path.exists(self._get_model_path() + ".zip")
    
    def previous_exists(self):
        prev_path = self._get_previous_model_path()
        if prev_path is None:
            return False
        return os.path.exists(prev_path + ".zip")
    
    def train(self, env, timesteps, continue_from=None, checkpoint_freq=50000):
        """
        train the agent.
        """
        # wrap environment for stable-baselines3
        monitored_env = Monitor(env, self.log_dir)
        vec_env = DummyVecEnv([lambda: monitored_env])
        
        # determine what model to start from
        if continue_from is not None:
            # Explicit path loading
            if not os.path.exists(continue_from + ".zip"):
                raise FileNotFoundError(f"model not found: {continue_from}.zip")
            print(f"loading model from: {continue_from}")
            self.model = PPO.load(continue_from, env=vec_env, device=self.device)
            
        elif self.level > 0 and self.previous_exists():
            # Auto-load previous level
            prev_path = self._get_previous_model_path()
            print(f"loading previous level model: {prev_path}")
            self.model = PPO.load(prev_path, env=vec_env, device=self.device)
            
        elif self.level > 0 and not self.previous_exists():
            # Warn and create fresh
            print(f"warning: level {self.level - 1} model not found")
            print("creating fresh model instead (not recommended for level > 0)")
            self.model = self._create_fresh_model(vec_env)
            
        else:
            # Level 0
            print("creating fresh model")
            self.model = self._create_fresh_model(vec_env)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix=self._get_model_name()
        )
        
        print(f"starting training for {timesteps} timesteps...")
        print(f"  level: {self.level}")
        print(f"  walls variant: {self.walls_variant}")
        print(f"  learning rate: {self.learning_rate}")
        print(f"  device: {self.device}") 
        
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
            tensorboard_log=self.log_dir,
            device=self.device 
        )
    
    def save(self):
        if self.model is None:
            raise RuntimeError("no model to save. call train() or load() first")
        
        save_path = self._get_model_path()
        self.model.save(save_path)
        print(f"model saved to: {save_path}.zip")
        return save_path
    
    def load(self, path=None):
        if path is None:
            path = self._get_model_path()
        
        if not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"model not found: {path}.zip")
        
        self.model = PPO.load(path, device=self.device)
        print(f"model loaded from: {path}.zip (on {self.device})")
        return self
    
    def predict(self, obs, deterministic=True):
        if self.model is None:
            raise RuntimeError("no model loaded. call train() or load() first")
        
        return self.model.predict(obs, deterministic=deterministic)


def get_agent_defaults():
    return {
        "level": 0,
        "walls_variant": False,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "verbose": 1,
        "timesteps": 500000,
        "continue_from": None,
        "device": "auto"
    }


def get_agent_options():
    return {
        "level": {
            "type": "int",
            "range": [0, 99],
            "description": "agent version number"
        },
        "walls_variant": {
            "type": "bool",
            "choices": [True, False],
            "descriptions": {True: "with walls", False: "no walls"}
        },
        "learning_rate": {
            "type": "float",
            "range": [1e-6, 1e-1],
            "description": "ppo learning rate"
        },
        "n_steps": {
            "type": "int",
            "range": [64, 8192],
            "description": "steps collected before update"
        },
        "batch_size": {
            "type": "int",
            "range": [8, 512],
            "description": "mini-batch size"
        },
        "verbose": {
            "type": "int",
            "choices": [0, 1, 2],
            "descriptions": {0: "none", 1: "info", 2: "debug"}
        },
        "timesteps": {
            "type": "int",
            "range": [1000, 10000000],
            "description": "total training timesteps"
        },
        "continue_from": {
            "type": "str",
            "description": "model path or 'auto'"
        },
        "device": {
            "type": "str",
            "choices": ["auto", "cpu", "cuda", "mps"],
            "descriptions": {
                "auto": "automatic detection",
                "cpu": "force cpu",
                "cuda": "nvidia gpu",
                "mps": "mac metal (apple silicon)"
            }
        }
    }