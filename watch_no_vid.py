# --- THIS SCRIPT LETS YOU WATCH THE TRAINED AI PLAY THE GAME --- #
# --- NO VIDEO RECORDING, JUST REAL-TIME RENDERING TO THE SCREEN --- #


import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

# Import your game
from tank_combat_env import TankCombatEnv

# ==========================================
# 📺 CONFIGURATION
# ==========================================
# Path to your saved model (without the .zip extension) --- ADJUST AS NEEDED
MODEL_PATH = "checkpoints/tank_run_550000_steps"

# How many matches do you want to watch? ---- ADJUST AS NEEDED
NUM_MATCHES = 10
# ==========================================

print("Loading the environment...")
# We set render_mode="human" to pop up the game window
env = TankCombatEnv(render_mode="human")

print(f"Loading model from: {MODEL_PATH}...")
# Load the trained brain
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH)
else:
    print(f"❌ Error: Could not find model at {MODEL_PATH}.zip")
    exit()

print("---------------------------------------")
print("👀 WATCHING AI PLAY (Press Ctrl+C to stop)")
print("---------------------------------------")

for episode in range(NUM_MATCHES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    score = 0
    
    print(f"Match {episode + 1} Started...")
    
    while not (terminated or truncated):
        # IMPORTANT: deterministic=True means "Use your BEST move"
        # deterministic=False means "Use random moves sometimes to learn"
        # Since we are watching, we want it to play its best!
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        
        # Slow down slightly so our human eyes can see what's happening
        # (Remove this line if you want to see it play at 1000x speed)
        env.render()
        time.sleep(0.01) 

    print(f"Match Finished! Total Score: {score:.2f}")

env.close()