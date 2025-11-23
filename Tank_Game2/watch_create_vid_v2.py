# --- THIS SCRIPT RECORDS A SINGLE CONTINUOUS VIDEO OF 5 MATCHES --- #

import gymnasium as gym
import numpy as np
import os
import imageio
from stable_baselines3 import PPO

# Import your game
from tank_combat_env import TankCombatEnv

# ==========================================
# 📺 CONFIGURATION
# ==========================================
MODEL_PATH = "models/tank_ai_final_pro_v2"
VIDEO_FOLDER = "tank_recordings_v2" 
FILENAME = "tank_v2_montage.mp4"
NUM_MATCHES = 5 
FPS = 60 # The video playback speed
# ==========================================

print("Loading the environment...")
os.makedirs(VIDEO_FOLDER, exist_ok=True)
video_path = os.path.join(VIDEO_FOLDER, FILENAME)

# Render mode must be 'rgb_array' to get pixel data
env = TankCombatEnv(render_mode="rgb_array")

# !!! CRITICAL: Set difficulty to 3 (PRO) !!!
env.unwrapped.bot_difficulty = 3

print(f"Loading model from: {MODEL_PATH}...")
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH, device="cpu")
else:
    print(f"❌ Error: Could not find model at {MODEL_PATH}.zip")
    exit()

print("---------------------------------------")
print(f"🎥 RECORDING CONTINUOUS VIDEO")
print(f"   Episodes: {NUM_MATCHES}")
print(f"   Output:   {video_path}")
print("---------------------------------------")

# Create a Video Writer
try:
    writer = imageio.get_writer(video_path, fps=FPS)
except ImportError:
    print("❌ Error: 'imageio' or 'ffmpeg' not found.")
    print("   Try running: pip install imageio imageio[ffmpeg]")
    exit()

total_score = 0

for episode in range(NUM_MATCHES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    episode_score = 0
    
    print(f"Recording Match {episode + 1}/{NUM_MATCHES}...")
    
    # Capture the spawn frame
    first_frame = env.render()
    writer.append_data(first_frame)
    
    while not (terminated or truncated):
        # 1. Get Action (Deterministic = Best Move)
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. Step Environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_score += reward
        
        # 3. Capture Frame
        # env.render() returns a numpy array of the screen pixels
        frame = env.render() 
        writer.append_data(frame)

    print(f"   Match Finished! Score: {episode_score:.2f}")
    total_score += episode_score

# Close everything to save the file
writer.close()
env.close()

print("---------------------------------------")
print(f"✅ DONE! Video saved to: {video_path}")
print(f"   Average Score: {total_score / NUM_MATCHES:.2f}")
print("---------------------------------------")