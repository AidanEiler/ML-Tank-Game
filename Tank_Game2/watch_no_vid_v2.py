# --- WATCH THE TRAINED AI VS THE PRO BOT --- #

import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Import your game
from tank_combat_env import TankCombatEnv

# ==========================================
# 📺 CONFIGURATION
# ==========================================
# 1. CHANGE THIS to your new V2 model
MODEL_PATH = "models/tank_ai_final_pro_v2" 

# How many matches do you want to watch?
NUM_MATCHES = 5
# ==========================================

print("Loading the environment...")
env = TankCombatEnv(render_mode="human")

# 2. CHANGE THIS to 3 (Pro Mode)
# Your V2 model is too smart for Level 0. Let's see it fight the Boss.
env.unwrapped.bot_difficulty = 3

print(f"Loading model from: {MODEL_PATH}...")

if os.path.exists(MODEL_PATH + ".zip"):
    # Force CPU to avoid the Nvidia crash
    model = PPO.load(MODEL_PATH, device="cpu")
else:
    print(f"❌ Error: Could not find model at {MODEL_PATH}.zip")
    exit()

print("---------------------------------------")
print("👀 WATCHING AI vs LEVEL 3 PRO BOT")
print("---------------------------------------")

for episode in range(NUM_MATCHES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    score = 0
    
    print(f"Match {episode + 1} Started...")
    
    while not (terminated or truncated):
        # deterministic=True makes the AI play its absolute best move
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # The new env renders at correct FPS automatically, 
        # but if you want to slow it down to Matrix-style slow mo, uncomment this:
        # time.sleep(0.02) 

    print(f"Match Finished! Total Score: {score:.2f}")
    
    if score > 2.0:
        print(">>> DOMINATION (Ready for Self-Play)")
    elif score > 0:
        print(">>> CLOSE FIGHT")
    else:
        print(">>> LOSS")

env.close()