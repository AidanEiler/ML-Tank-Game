"""
play_human.py
"""
import pygame
import numpy as np
from tank_combat_env import TankCombatEnv

# Initialize environment
env = TankCombatEnv(render_mode="human", opponent_type="bot")

# Set the bot diffculty when playing human vs bot here.
env.unwrapped.bot_difficulty = 1

obs, info = env.reset()
env.render()

running = True
wins, losses = 0, 0

print("=" * 50)
print("🎮 TANK COMBAT - VS PRO BOT")
print("=" * 50)
print("You are the BLUE TANK.")
print("The GREEN LINES are your Lidar Sensors (The AI sees these!).")
print("-" * 50)
print("DRIVING (WASD):")
print("  W + D = Up-Right Diagonal (Supported!)")
print("-" * 50)
print("COMBAT (Arrows + Space):")
print("  LEFT / RIGHT: Rotate Turret")
print("  SPACE       : Fire")
print("-" * 50)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    keys = pygame.key.get_pressed()
    
    # Action: [Body_X, Body_Y, Turret, Fire]
    action = np.array([0, 0, 0, 0]) 
    
    # -- Horizontal (Body X) --
    if keys[pygame.K_a]:
        action[0] = 1 # Left
    elif keys[pygame.K_d]:
        action[0] = 2 # Right
        
    # -- Vertical (Body Y) --
    if keys[pygame.K_w]:
        action[1] = 1 # Up
    elif keys[pygame.K_s]:
        action[1] = 2 # Down
        
    # -- Turret --
    if keys[pygame.K_LEFT]:
        action[2] = 1 # Left
    elif keys[pygame.K_RIGHT]:
        action[2] = 2 # Right
        
    # -- Fire --
    if keys[pygame.K_SPACE]:
        action[3] = 1
        
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render() handles the FPS clock tick
    env.render()
    
    if terminated or truncated:
        if reward > 0:
            wins += 1
            print(f"🎉 YOU WIN! Score: {wins}-{losses}")
        else:
            losses += 1
            print(f"💀 YOU LOSE! Score: {wins}-{losses}")
            
        obs, info = env.reset()

env.close()
print(f"\nFinal Score: {wins} Wins - {losses} Losses")