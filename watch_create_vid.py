# --- THIS SCRIPT RECORDS VIDEOS OF THE TRAINED AI PLAYING THE GAME --- #


import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import os

# Import your game
from tank_combat_env import TankCombatEnv

# ==========================================
# 📺 CONFIGURATION
# ==========================================
MODEL_PATH = "models/tank_ai_level_0_finished"  # <--- Path to your saved model to record from
NUM_MATCHES = 5  # Keep this low for video generation
VIDEO_FOLDER = "tank_recordings" # Where to save the videos
# ==========================================

print("Loading the environment...")

# This tells the game: "Don't open a window, just give me the pixel data."
env = TankCombatEnv(render_mode="rgb_array")

# 2. Wrap the environment to record video
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    name_prefix="tank_ai_gameplay",
    # This lambda function determines which episodes to record.
    # "return True" means "Record EVERY episode"
    episode_trigger=lambda episode_id: True 
)

print(f"Loading model from: {MODEL_PATH}...")
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(MODEL_PATH)
else:
    print(f"❌ Error: Could not find model at {MODEL_PATH}.zip")
    exit()

print("---------------------------------------")
print(f"🎥 RECORDING AI (Check the '{VIDEO_FOLDER}' folder after)")
print("---------------------------------------")

for episode in range(NUM_MATCHES):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    score = 0
    
    print(f"Recording Match {episode + 1}...")
    
    while not (terminated or truncated):
        # deterministic=True for best performance
        action, _ = model.predict(obs, deterministic=True)
        
        # The wrapper automatically captures the frame here
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

    print(f"Match Finished! Total Score: {score:.2f}")

# 3. Important: Close the env to ensure the video file is saved properly
env.close()
print(f"✅ Done! Videos saved in {os.path.abspath(VIDEO_FOLDER)}")