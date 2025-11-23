import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

# IMPORT YOUR CUSTOM ENV
# Ensure your environment file is named 'tank_combat_env.py'
from tank_combat_env import TankCombatEnv 

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
LOG_DIR = "./tank_tensorboard/"
MODELS_DIR = "./models/"
CHECKPOINT_DIR = "./checkpoints/"
TOTAL_TIMESTEPS = 3_000_000  # Set high (3M). The agent determines the pace, not the clock.

# 🎯 PROMOTION THRESHOLDS (Avg Reward over 100 episodes)
# Level 0 -> 1: Needs > 3.30 (Must hit stationary target perfectly)
# Level 1 -> 2: Needs > 3.5 (Must hit linearly moving target consistently)
# Level 2 -> 3: Needs > 2.0 (Must win > 50% of fights against Smart Bot)
DIFFICULTY_THRESHOLDS = {
    0: 3.30,
    1: 3.5,
    2: 2.0,
    3: 999.0  # Max level, never upgrade
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 🧠 CURRICULUM MANAGER (The Brain)
# ==========================================
class CurriculumCallback(BaseCallback):
    """
    Monitors training performance. 
    If the agent consistently beats the current difficulty, it levels up.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.current_difficulty = 0

    def _on_step(self) -> bool:
        # Only check every X steps to save processing time
        if self.n_calls % self.check_freq == 0:

            # 1. Load Data
            # 'x' is timesteps, 'y' is rewards
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            
            # 2. SAFETY CHECK: Minimum Data Requirement
            # If we haven't played 100 episodes yet, don't do anything.
            # This prevents lucky streaks at the very start from triggering a promotion.
            if len(x) < 100:
                return True

            # 3. Calculate Average of Last 100 Episodes
            mean_reward = np.mean(y[-100:])
            
            # Get current difficulty from the environment
            # We access the 'unwrapped' env to get to the custom variables
            current_diff = self.training_env.envs[0].unwrapped.bot_difficulty
            
            # Sync local variable just in case
            if current_diff != self.current_difficulty:
                self.current_difficulty = current_diff

            if self.verbose > 0:
                print(f"Step: {self.num_timesteps} | Diff: {current_diff} | Mean Reward (100 eps): {mean_reward:.2f}")

            # 4. Check for Promotion
            required_score = DIFFICULTY_THRESHOLDS.get(current_diff, 999)
            
            if mean_reward > required_score:
                if current_diff < 3: # If not at max level
                    new_diff = current_diff + 1
                    
                    print("\n" + "="*50)
                    print(f"🎉 PROMOTION TRIGGERED!")
                    print(f"   Current Level: {current_diff}")
                    print(f"   Avg Score:     {mean_reward:.2f} (Threshold: {required_score})")
                    print(f"   🚀 Moving to Level {new_diff}...")
                    print("="*50 + "\n")
                    
                    # A. Save the "Graduation" Model
                    save_name = os.path.join(MODELS_DIR, f"tank_graduate_lvl{current_diff}")
                    self.model.save(save_name)
                    print(f"   Saved checkpoint: {save_name}")

                    # B. Update the Environment Difficulty
                    self.training_env.envs[0].unwrapped.bot_difficulty = new_diff
                    self.current_difficulty = new_diff
                    
                else:
                    # Already at Max Level
                    pass 
            
        return True

# ==========================================
# 🚀 MAIN TRAINING LOOP
# ==========================================

# 1. Setup Environment
# We use Monitor to log stats for TensorBoard and our Callback
env = TankCombatEnv(render_mode=None)
env = Monitor(env, LOG_DIR) 
env = DummyVecEnv([lambda: env])

# Ensure we start at Difficulty 0
env.envs[0].unwrapped.bot_difficulty = 0

# 2. Setup PPO Model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log=LOG_DIR,
    device="cpu"
)

# 3. Setup Callback
# Check progress every 10,000 steps.
curriculum_callback = CurriculumCallback(check_freq=10000, log_dir=LOG_DIR)

print("--------------------------------------------------")
print("STARTING ADAPTIVE TRAINING SESSION")
print(f"Logs: {LOG_DIR}")
print(f"Models: {MODELS_DIR}")
print("The agent will auto-promote when it proves it is ready.")
print("--------------------------------------------------")

# 4. Train
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=curriculum_callback)
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted manually.")
    model.save(os.path.join(MODELS_DIR, "tank_ai_interrupted"))

# 5. Final Save
final_path = os.path.join(MODELS_DIR, "tank_ai_final_pro")
model.save(final_path)

print("\n=== Training Complete ===")
print(f"Final model saved to: {final_path}.zip")