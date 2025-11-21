# --- Secondary training Script for Level 1 AI, loads Level 0 model and continues training --- #
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from tank_combat_env import TankCombatEnv 

# Directories
log_dir = "./tank_tensorboard/"
checkpoint_dir = "./checkpoints/"
models_dir = "./models/"

# 1. Create environment (Make sure tank_env.py has BOT_DIFFICULTY = 1)
env = TankCombatEnv(render_mode=None)
env = Monitor(env, log_dir) 
env = DummyVecEnv([lambda: env])

# 2. LOAD THE LEVEL 0 BRAIN
# We are picking up where we left off!
print("Loading Level 0 Model...")
model_path = "models_level_0/tank_ai_level_0_finished" # The file you just made
model = PPO.load(model_path, env=env)

# 3. Create Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=50000, 
    save_path=checkpoint_dir, 
    name_prefix='tank_level_1_run'
)

print("--------------------------------------------------")
print("STARTING TRAINING... (Level 1 - The Noob)")
print("The AI should struggle at first, then adapt.")
print("--------------------------------------------------")

# 4. Train for another 500k steps
# The tensorboard log will continue as "PPO_Level_1"
model.learn(total_timesteps=500000, callback=checkpoint_callback, tb_log_name="PPO_Level_1")

# 5. Save Level 1 Model
save_path = os.path.join(models_dir, "tank_ai_level_1_finished")
model.save(save_path)

print("\n=== Level 1 Complete ===")
print(f"Model saved at: {save_path}.zip")