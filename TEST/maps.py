"""
maps.py - Advanced tank combat map generator
"""

import pygame
import numpy as np

def get_map_layout(style, grid_size):
    """
    Returns a list of pygame.Rect objects representing walls.
    """
    obstacles = []
    mid = grid_size // 2
    
    # Define Spawn Safety Zones (Top-Left and Bottom-Right)
    # No walls allowed within 150px of start points
    def is_in_spawn(rect):
        spawn_zone = 150
        # Check Top-Left Spawn
        if rect.left < spawn_zone and rect.top < spawn_zone: return True
        # Check Bottom-Right Spawn
        if rect.right > grid_size - spawn_zone and rect.bottom > grid_size - spawn_zone: return True
        return False

    # --- 1. CLASSIC (Original Training Map) ---
    if style == "Classic":
        obstacles = [
            pygame.Rect(mid - 40, mid - 40, 80, 80),
            pygame.Rect(100, 100, 40, 40),
            pygame.Rect(grid_size - 140, 100, 40, 40),
            pygame.Rect(100, grid_size - 140, 40, 40),
            pygame.Rect(grid_size - 140, grid_size - 140, 40, 40),
            pygame.Rect(mid - 100, 100, 200, 60), 
            pygame.Rect(mid - 100, grid_size - 160, 200, 60) 
        ]

    # --- 2. EMPTY (Open field) ---
    elif style == "Empty":
        obstacles = []

    # --- 3. DYNAMIC (Smart Random Blocks) ---
    elif style == "Dynamic":
        num_walls = np.random.randint(6, 12)
        attempts = 0
        
        while len(obstacles) < num_walls and attempts < 100:
            attempts += 1
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            x = np.random.randint(0, grid_size - w)
            y = np.random.randint(0, grid_size - h)
            
            new_wall = pygame.Rect(x, y, w, h)
            
            # Check 1: Is it in a spawn zone?
            if is_in_spawn(new_wall): continue
                
            # Check 2: Does it overlap with existing walls?
            # We inflate the rect slightly to ensure gaps between walls
            if new_wall.inflate(20, 20).collidelist(obstacles) != -1:
                continue
                
            obstacles.append(new_wall)

    # --- 4. MAZE (City Grid Style) ---
    elif style == "Maze":
        # Create a grid of city blocks
        block_size = 80
        street_width = 60
        
        step = block_size + street_width
        
        for x in range(street_width, grid_size - street_width, step):
            for y in range(street_width, grid_size - street_width, step):
                
                # 20% chance to skip a block (creates open plazas)
                if np.random.random() > 0.2:
                    wall = pygame.Rect(x, y, block_size, block_size)
                    
                    # Don't block spawns
                    if not is_in_spawn(wall):
                        obstacles.append(wall)

    # --- 5. FOREST (Scattered Small Cover) ---
    elif style == "Forest":
        num_trees = 40
        for _ in range(num_trees):
            size = 30
            x = np.random.randint(0, grid_size - size)
            y = np.random.randint(0, grid_size - size)
            tree = pygame.Rect(x, y, size, size)
            
            if not is_in_spawn(tree):
                obstacles.append(tree)

    return obstacles