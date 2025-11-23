import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Import Environment
from tank_combat_env import TankCombatEnv 

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 1. Load from this file
LOAD_MODEL_PATH = "models/tank_ai_final_pro"  

# 2. Save Final Result to this file
SAVE_MODEL_PATH = "models/tank_ai_final_pro_v2"

# 3. Starting Difficulty
START_DIFFICULTY = 2 

LOG_DIR = "./tank_tensorboard/"
MODELS_DIR = "./models/"
CHECKPOINT_DIR = "./checkpoints/"
TOTAL_TIMESTEPS = 10_000_000 

# Thresholds (We only care about 2 -> 3 now)
DIFFICULTY_THRESHOLDS = {
    0: 3.30, 
    1: 1.25,
    2: 1.50,   # Needs > 2.0 to beat Level 2
    3: 999.0  # Max level
}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 🧠 CURRICULUM MANAGER
# ==========================================
class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.current_difficulty = START_DIFFICULTY # Start tracking at Level 2

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            
            if len(x) < 100: return True

            mean_reward = np.mean(y[-100:])
            current_diff = self.training_env.envs[0].unwrapped.bot_difficulty
            
            if current_diff != self.current_difficulty:
                self.current_difficulty = current_diff

            if self.verbose > 0:
                print(f"Step: {self.num_timesteps} | Diff: {current_diff} | Mean Reward (100 eps): {mean_reward:.2f}")

            required_score = DIFFICULTY_THRESHOLDS.get(current_diff, 999)
            
            if mean_reward > required_score:
                if current_diff < 3:
                    new_diff = current_diff + 1
                    print("\n" + "="*50)
                    print(f"🎉 PROMOTION TRIGGERED!")
                    print(f"   Current Level: {current_diff}")
                    print(f"   Avg Score:     {mean_reward:.2f} (Threshold: {required_score})")
                    print(f"   🚀 Moving to Level {new_diff}...")
                    print("="*50 + "\n")
                    
                    save_name = os.path.join(MODELS_DIR, f"tank_graduate_lvl{current_diff}")
                    self.model.save(save_name)
                    print(f"   Saved checkpoint: {save_name}")

                    self.training_env.envs[0].unwrapped.bot_difficulty = new_diff
                    self.current_difficulty = new_diff
            
        return True

# ==========================================
# 🚀 RESUME TRAINING LOOP
# ==========================================

# 1. Setup Environment
env = TankCombatEnv(render_mode=None)
env = Monitor(env, LOG_DIR) 
env = DummyVecEnv([lambda: env])

# 2. Force Start at Level 2
print(f"Setting Environment to Difficulty Level {START_DIFFICULTY}...")
env.envs[0].unwrapped.bot_difficulty = START_DIFFICULTY

# 3. Load Existing Model
print(f"Loading model weights from: {LOAD_MODEL_PATH}.zip")
if not os.path.exists(LOAD_MODEL_PATH + ".zip"):
    print("❌ ERROR: Model file not found! Check path.")
    exit()

# Note: We use device="cpu" to match your hardware setup
model = PPO.load(LOAD_MODEL_PATH, env=env, device="cpu")

# 4. Setup Callbacks
curriculum_callback = CurriculumCallback(check_freq=10000, log_dir=LOG_DIR)
checkpoint_callback = CheckpointCallback(
    save_freq=200000, 
    save_path=CHECKPOINT_DIR, 
    name_prefix="tank_v2_backup" # Different prefix for V2 backups
)
callback_list = CallbackList([curriculum_callback, checkpoint_callback])

print("--------------------------------------------------")
print("🚀 RESUMING TRAINING (V2)")
print(f"Starting Level: {START_DIFFICULTY}")
print(f"Target Steps:   {TOTAL_TIMESTEPS}")
print("--------------------------------------------------")

# 5. Train
try:
    # reset_num_timesteps=False ensures Tensorboard continues the graph line
    # instead of starting over at 0.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list, reset_num_timesteps=False)
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted manually.")
    model.save(os.path.join(MODELS_DIR, "tank_ai_v2_interrupted"))

# 6. Final Save
model.save(SAVE_MODEL_PATH)

print("\n=== Training Complete ===")
print(f"Final model saved to: {SAVE_MODEL_PATH}.zip")