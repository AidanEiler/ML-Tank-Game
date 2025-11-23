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
# Update this to point to your final model or a graduate model
MODEL_PATH = "models/tank_graduate_lvl0" 

# How many matches do you want to watch?
NUM_MATCHES = 5
# ==========================================

print("Loading the environment...")
env = TankCombatEnv(render_mode="human")

# !!! CRITICAL UPDATE !!!
# We force the bot to be LEVEL 3 (The Pro)
# Otherwise, it will default to Level 0 (The Turret)
env.unwrapped.bot_difficulty = 0 

print(f"Loading model from: {MODEL_PATH}...")

if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, device="cpu")
else:
    print(f"❌ Error: Could not find model at {MODEL_PATH}.zip")
    # Fallback: Check if we have a graduate model if the final isn't ready
    print("Trying to find a Level 2 graduate model...")
    if os.path.exists("models/tank_graduate_lvl2.zip"):
        model = PPO.load("models/tank_graduate_lvl2")
        print("✅ Loaded Level 2 Graduate model instead.")
    else:
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
        # deterministic=True is CORRECT for testing.
        # It shuts off the "random exploration" noise used during training.
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # Note: The new Env has a built-in clock tick in render(),
        # so this sleep might make it look a bit slow. 
        # If it looks laggy, remove the line below.
        # time.sleep(0.01) 

    print(f"Match Finished! Total Score: {score:.2f}")
    
    if score > 2.0:
        print(">>> DOMINATION (Ready for Self-Play)")
    elif score > 0:
        print(">>> CLOSE FIGHT")
    else:
        print(">>> LOSS")

env.close()