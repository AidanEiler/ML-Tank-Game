import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from tank_combat_env import TankCombatEnv 

# ==========================================
# 📁 FOLDER SETUP
# ==========================================
log_dir = "./tank_tensorboard/"
checkpoint_dir = "./checkpoints/"
models_dir = "./models/" # folder to save final models

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True) # <--- Creates the folder if it doesn't exist
# ==========================================

# 1. Create environment
# Ensure 'tank_combat_env.py' has BOT_DIFFICULTY = 0 for this first run!
env = TankCombatEnv(render_mode=None)
env = Monitor(env, log_dir) 
env = DummyVecEnv([lambda: env])

# 2. Create Checkpoint Callback
# Saves backup copies to ./checkpoints/ every 50k steps
checkpoint_callback = CheckpointCallback(
    save_freq=50000, 
    save_path=checkpoint_dir, 
    name_prefix='tank_run'
)

# 3. Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    tensorboard_log=log_dir
)

print("--------------------------------------------------")
print("STARTING TRAINING... (Level 0 - The Zombie)")
print(f"Backups will save to: {checkpoint_dir}")
print(f"Final model will save to: {models_dir}")
print("--------------------------------------------------")

# 4. Train
try:
    model.learn(total_timesteps=800000, callback=checkpoint_callback, tb_log_name="PPO_Level_0")
except KeyboardInterrupt:
    print("\nTraining interrupted manually.")

# 5. Save Final Model to the 'models/' folder
save_path = os.path.join(models_dir, "tank_ai_level_0_finished")
model.save(save_path)

print("\n=== Training Complete ===")
print(f"Model saved at: {save_path}.zip")