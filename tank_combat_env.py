import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

# ==========================================
# ⚙️ TRAINING CURRICULUM SETTINGS
# ==========================================
USE_WALLS = False

# 0 = ZOMBIE (Moves slowly towards you, stops when close. Bad aim.)
# 1 = NOOB (Normal speed, but gets stuck on walls. Bad aim.)
# 2 = AVERAGE (Decent aim, standard movement, slight delay)
# 3 = PRO (Perfect aim, slides around corners, instant fire)
BOT_DIFFICULTY = 3
# ==========================================

class HeuristicOpponent:
    def __init__(self, use_walls=True):
        self.move_timer = 0
        self.current_action = [0, 0, 0, 0]
        
        # --- RANDOMIZATION STATE ---
        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)

        # --- MAP DATA ---
        self.grid_size = 600
        mid = self.grid_size // 2
        
        if use_walls:
            self.obstacles = [
                pygame.Rect(mid - 40, mid - 40, 80, 80),
                pygame.Rect(100, 100, 40, 40),
                pygame.Rect(self.grid_size - 140, 100, 40, 40),
                pygame.Rect(100, self.grid_size - 140, 40, 40),
                pygame.Rect(self.grid_size - 140, self.grid_size - 140, 40, 40),
                pygame.Rect(mid - 100, 100, 200, 60), 
                pygame.Rect(mid - 100, self.grid_size - 160, 200, 60)
            ]
        else:
            self.obstacles = []

    def reset_episode(self):
        self.move_timer = 0
        self.current_action = [0, 0, 0, 0]
        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)

    def predict(self, obs, deterministic=True):
        # === 1. CHAOS (Exploration) ===
        # Level 0 acts random 10% of the time to seem confused
        chaos_thresh = 0.10 if BOT_DIFFICULTY <= 1 else 0.05
        
        if np.random.random() < chaos_thresh:
            rand_act = np.array([
                np.random.randint(0, 3), 
                np.random.randint(0, 3), 
                np.random.randint(0, 3), 
                np.random.randint(0, 2)
            ])
            return rand_act, None

        # Extract Data
        norm_me_x, norm_me_y = obs[6], obs[7]
        norm_enemy_x, norm_enemy_y = obs[0], obs[1]
        
        me_turret_sin, me_turret_cos = obs[10], obs[11]
        me_turret_angle = math.degrees(math.atan2(me_turret_sin, me_turret_cos)) % 360
        
        cooldown = obs[19] 
        
        # Calculations
        dx = norm_enemy_x - norm_me_x
        dy = norm_enemy_y - norm_me_y
        dist = math.sqrt(dx**2 + dy**2)
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        turret_diff = (target_angle - me_turret_angle + 180) % 360 - 180
        
        # Pixel Positions
        pix_me_x, pix_me_y = float(norm_me_x * self.grid_size), float(norm_me_y * self.grid_size)
        pix_enemy_x, pix_enemy_y = float(norm_enemy_x * self.grid_size), float(norm_enemy_y * self.grid_size)

        # --- LINE OF SIGHT CHECK ---
        has_los = True
        for wall in self.obstacles:
            if wall.clipline(pix_me_x, pix_me_y, pix_enemy_x, pix_enemy_y):
                has_los = False
                break

        # --- BARREL CLEARANCE CHECK ---
        barrel_rad = math.radians(me_turret_angle)
        barrel_x = pix_me_x + math.cos(barrel_rad) * 35
        barrel_y = pix_me_y + math.sin(barrel_rad) * 35
        
        barrel_blocked = False
        for wall in self.obstacles:
            if wall.collidepoint(barrel_x, barrel_y):
                barrel_blocked = True
                break
        
        # === DECISION LOGIC ===
        actions = [0, 0, 0, 0] # [Body_X, Body_Y, Turret, Fire]

        # 1. Aim (With Difficulty Jitter)
        # Level 0: Terrible aim (+/- 40 degrees)
        aim_margin = 40 if BOT_DIFFICULTY == 0 else (20 if BOT_DIFFICULTY == 1 else (10 if BOT_DIFFICULTY == 2 else 5))
        
        if abs(turret_diff) > aim_margin:
            if turret_diff < 0: actions[2] = 2 
            else: actions[2] = 1 
            
        # 2. Fire (With Reaction Delay)
        # Level 0: 1% chance (Basically never shoots)
        fire_prob = 0.01 if BOT_DIFFICULTY == 0 else (0.2 if BOT_DIFFICULTY == 1 else (0.5 if BOT_DIFFICULTY == 2 else 1.0))
        
        if has_los and not barrel_blocked and abs(turret_diff) < 15 and cooldown < 0.1:
            if np.random.random() < fire_prob:
                actions[3] = 1 

        # 3. MOVEMENT
        def is_move_safe(ax, ay):
            vx, vy = 0, 0
            if ax == 1: vx = -40 
            elif ax == 2: vx = 40 
            if ay == 1: vy = -40 
            elif ay == 2: vy = 40
            
            future_x = pix_me_x + vx
            future_y = pix_me_y + vy
            
            if future_x < 20 or future_x > self.grid_size - 20 or future_y < 20 or future_y > self.grid_size - 20:
                return False
            
            future_rect = pygame.Rect(future_x - 20, future_y - 20, 40, 40)
            if future_rect.collidelist(self.obstacles) != -1:
                return False
            return True

        # Check if CURRENT plan is still valid
        plan_valid = True
        if self.move_timer > 0:
            if not is_move_safe(self.current_action[0], self.current_action[1]):
                plan_valid = False
                self.move_timer = 0
        
        self.move_timer -= 1
        if self.move_timer > 0 and plan_valid:
            actions[0] = self.current_action[0]
            actions[1] = self.current_action[1]
            return np.array(actions), None

        # --- PICK NEW MOVE ---
        if BOT_DIFFICULTY == 0:
            self.move_timer = 5 # Change moves very often (to support the stutter step)
        elif BOT_DIFFICULTY == 1:
            self.move_timer = np.random.randint(30, 60)
        else:
            self.move_timer = np.random.randint(5, 25)

        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)
        
        # Strategy Selection
        ideal_ax, ideal_ay = 0, 0
        
        # === LEVEL 0 LOGIC: THE ZOMBIE ===
        if BOT_DIFFICULTY == 0:
            # Move towards enemy (Zombie walk)
            ideal_ax = 2 if dx > 0 else 1
            ideal_ay = 2 if dy > 0 else 1
            
            # 1. STOP if close enough (Don't run into the player)
            if dist < 0.4:
                ideal_ax, ideal_ay = 0, 0
            
            # 2. STUTTER STEP (Slow down)
            # Only register the move 30% of the time. 70% of the time, stand still.
            elif np.random.random() > 0.3:
                ideal_ax, ideal_ay = 0, 0
        
        # === LEVEL 1-3 LOGIC ===
        else:
            if not has_los:
                if abs(dx) > abs(dy): 
                    ideal_ay = 2 if dy > 0 else 1
                else:
                    ideal_ax = 2 if dx > 0 else 1
            else:
                if dist < self.retreat_dist: 
                    ideal_ax = 1 if dx > 0 else 2
                    ideal_ay = 1 if dy > 0 else 2
                elif dist > self.chase_dist: 
                    ideal_ax = 2 if dx > 0.05 else (1 if dx < -0.05 else 0)
                    ideal_ay = 2 if dy > 0.05 else (1 if dy < -0.05 else 0)
                else: 
                    p_dx, p_dy = -dy, dx
                    ideal_ax = 2 if p_dx > 0 else 1
                    ideal_ay = 2 if p_dy > 0 else 1

        # --- COLLISION RESOLUTION ---
        if is_move_safe(ideal_ax, ideal_ay):
            actions[0], actions[1] = ideal_ax, ideal_ay
        else:
            # Only LEVEL 3 knows how to "Slide" and "Advance" around corners
            if BOT_DIFFICULTY == 3:
                adv_ax = 2 if dx > 0.05 else (1 if dx < -0.05 else 0)
                adv_ay = 2 if dy > 0.05 else (1 if dy < -0.05 else 0)
                
                if is_move_safe(adv_ax, adv_ay):
                    actions[0], actions[1] = adv_ax, adv_ay
                else:
                    if is_move_safe(ideal_ax, 0):
                        actions[0], actions[1] = ideal_ax, 0
                    elif is_move_safe(0, ideal_ay):
                        actions[0], actions[1] = 0, ideal_ay
                    else:
                        found_escape = False
                        for tx, ty in [(1,0), (2,0), (0,1), (0,2)]:
                             if is_move_safe(tx, ty):
                                 actions[0], actions[1] = tx, ty
                                 found_escape = True
                                 break
                        if not found_escape:
                             actions[0], actions[1] = 0, 0
            else:
                # Level 0, 1 & 2: If blocked, just stop.
                actions[0], actions[1] = 0, 0

        self.current_action = actions
        return np.array(actions), None

class TankCombatEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, grid_size=600, opponent_type="bot", opponent_policy=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.opponent_policy = opponent_policy
        
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 2])
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # Constants
        self.TANK_SPEED = 4.0
        self.ROT_SPEED_TURRET = 5.0 
        self.BULLET_SPEED = 12.0
        self.TANK_RADIUS = 20
        self.BULLET_RADIUS = 5
        self.MAX_STEPS = 3000
        self.COOLDOWN_MAX = 40
        
        # Tracking Previous Spawn Locations
        self.last_t1_pos = None
        self.last_t2_pos = None
        
        # Pass settings to bot
        self.heuristic_bot = HeuristicOpponent(use_walls=USE_WALLS)
        
        # Obstacles logic
        mid = self.grid_size // 2
        if USE_WALLS:
            self.obstacles = [
                pygame.Rect(mid - 40, mid - 40, 80, 80),
                pygame.Rect(100, 100, 40, 40),
                pygame.Rect(self.grid_size - 140, 100, 40, 40),
                pygame.Rect(100, self.grid_size - 140, 40, 40),
                pygame.Rect(self.grid_size - 140, self.grid_size - 140, 40, 40),
                pygame.Rect(mid - 100, 100, 200, 60), 
                pygame.Rect(mid - 100, self.grid_size - 160, 200, 60) 
            ]
        else:
            self.obstacles = []
        
        self.window = None
        self.clock = None
        self.font = None

    # --- UPDATED HELPER METHOD ---
    def _get_random_spawn(self, y_min, y_max, avoid_pos=None):
        """
        Finds a spawn point that is:
        1. Within the Y-zone.
        2. Not hitting a wall.
        3. At least 'min_dist' away from 'avoid_pos'.
        """
        padding = self.TANK_RADIUS + 5
        min_dist_from_last = 100 # 100 pixels away from last death spot
        
        for _ in range(100): 
            x = np.random.uniform(padding, self.grid_size - padding)
            y = np.random.uniform(y_min + padding, y_max - padding)
            candidate = np.array([x, y], dtype=np.float32)
            
            # 1. Check Distance Constraint (if there was a previous position)
            if avoid_pos is not None:
                dist = np.linalg.norm(candidate - avoid_pos)
                if dist < min_dist_from_last:
                    continue # Too close, pick another random spot
            
            # 2. Check Wall Constraint
            temp_rect = pygame.Rect(
                x - self.TANK_RADIUS, 
                y - self.TANK_RADIUS, 
                self.TANK_RADIUS * 2, 
                self.TANK_RADIUS * 2
            )
            
            if temp_rect.collidelist(self.obstacles) == -1:
                return candidate
        
        # Fallback: If we fail 100 times, just use center of zone
        return np.array([self.grid_size/2, (y_min + y_max)/2], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.opponent_type == "bot":
            self.heuristic_bot.reset_episode()
        
        quarter_map = self.grid_size * 0.25
        
        # --- TANK 1 (Player) ---
        # Pass the LAST position so we avoid it
        self.t1_pos = self._get_random_spawn(0, quarter_map, avoid_pos=self.last_t1_pos)
        self.t1_body_angle = 90 
        self.t1_turret_angle = 90
        self.t1_cooldown = 0
        
        # --- TANK 2 (Enemy) ---
        self.t2_pos = self._get_random_spawn(self.grid_size - quarter_map, self.grid_size, avoid_pos=self.last_t2_pos)
        self.t2_body_angle = 270 
        self.t2_turret_angle = 270
        self.t2_cooldown = 0
        
        # Save these positions for next time
        self.last_t1_pos = self.t1_pos.copy()
        self.last_t2_pos = self.t2_pos.copy()
        
        self.bullet1 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        self.bullet2 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        
        self.step_count = 0
        self.visual_effects = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        def norm_pos(val): return val / self.grid_size
        t1_b_rad = math.radians(self.t1_body_angle)
        t1_t_rad = math.radians(self.t1_turret_angle)
        t2_b_rad = math.radians(self.t2_body_angle)
        t2_t_rad = math.radians(self.t2_turret_angle)
        
        obs = np.array([
            norm_pos(self.t1_pos[0]), norm_pos(self.t1_pos[1]),
            math.sin(t1_b_rad), math.cos(t1_b_rad),
            math.sin(t1_t_rad), math.cos(t1_t_rad),
            norm_pos(self.t2_pos[0]), norm_pos(self.t2_pos[1]),
            math.sin(t2_b_rad), math.cos(t2_b_rad),
            math.sin(t2_t_rad), math.cos(t2_t_rad),
            norm_pos(self.bullet1["pos"][0]), norm_pos(self.bullet1["pos"][1]),
            1.0 if self.bullet1["active"] else 0.0,
            norm_pos(self.bullet2["pos"][0]), norm_pos(self.bullet2["pos"][1]),
            1.0 if self.bullet2["active"] else 0.0,
            self.t1_cooldown / self.COOLDOWN_MAX,
            self.t2_cooldown / self.COOLDOWN_MAX
        ], dtype=np.float32)
        return obs
    
    def _get_opponent_obs(self):
        def norm_pos(val): return val / self.grid_size
        t1_b_rad = math.radians(self.t1_body_angle)
        t1_t_rad = math.radians(self.t1_turret_angle)
        t2_b_rad = math.radians(self.t2_body_angle)
        t2_t_rad = math.radians(self.t2_turret_angle)
        
        obs = np.array([
            norm_pos(self.t2_pos[0]), norm_pos(self.t2_pos[1]),
            math.sin(t2_b_rad), math.cos(t2_b_rad),
            math.sin(t2_t_rad), math.cos(t2_t_rad),
            norm_pos(self.t1_pos[0]), norm_pos(self.t1_pos[1]),
            math.sin(t1_b_rad), math.cos(t1_b_rad),
            math.sin(t1_t_rad), math.cos(t1_t_rad),
            norm_pos(self.bullet2["pos"][0]), norm_pos(self.bullet2["pos"][1]),
            1.0 if self.bullet2["active"] else 0.0,
            norm_pos(self.bullet1["pos"][0]), norm_pos(self.bullet1["pos"][1]),
            1.0 if self.bullet1["active"] else 0.0,
            self.t2_cooldown / self.COOLDOWN_MAX,
            self.t1_cooldown / self.COOLDOWN_MAX
        ], dtype=np.float32)
        return obs

    def step(self, action):
            total_reward = 0
            terminated = False
            truncated = False
            
            # -------------------------------------------------
            # ⚡ FRAME SKIP LOGIC
            # -------------------------------------------------
            # If we are 'human' watching, or recording video ('rgb_array'), 
            # we might want to DISABLE skipping to see smooth motion.
            if self.render_mode in ["human", "rgb_array"]:
                repeat_frames = 1  # Don't skip! Smooth motion for eyes/cameras.
            else:
                repeat_frames = 4  # Skip! Fast speed for AI training.
            # -------------------------------------------------
            
            for _ in range(repeat_frames):
                
                # --- RENDERING (If Human Mode) ---
                if self.render_mode == "human":
                    self.render()
                # =============================================
                
                self.step_count += 1
                
                # 1. Predict Opponent Action
                if self.opponent_type == "bot":
                    op_obs = self._get_obs()
                    op_action, _ = self.heuristic_bot.predict(op_obs)
                elif self.opponent_type in ["self", "policy"] and self.opponent_policy:
                    op_obs = self._get_opponent_obs()
                    op_action, _ = self.opponent_policy.predict(op_obs, deterministic=False)
                else:
                    op_action = self.action_space.sample()
                
                # 2. Move Entities
                self._move_tank(1, action)
                self._move_tank(2, op_action)
                
                self._resolve_collisions()
                self._update_bullets()
                
                # 3. Accumulate Reward (Small penalty per frame)
                step_reward = -0.005
                
                if self.bullet1["active"]:
                    dist = np.linalg.norm(self.bullet1["pos"] - self.t2_pos)
                    if dist < self.TANK_RADIUS + self.BULLET_RADIUS:
                        step_reward += 5.0
                        terminated = True
                        self.bullet1["active"] = False 
                        self.visual_effects.append({"pos": self.t2_pos.copy(), "life": 15, "color": (255, 50, 50)})
                        
                if self.bullet2["active"]:
                    dist = np.linalg.norm(self.bullet2["pos"] - self.t1_pos)
                    if dist < self.TANK_RADIUS + self.BULLET_RADIUS:
                        step_reward -= 5.0
                        terminated = True
                        self.bullet2["active"] = False 
                        self.visual_effects.append({"pos": self.t1_pos.copy(), "life": 15, "color": (50, 150, 255)})

                # Check truncation (MAX_STEPS)
                if self.step_count >= self.MAX_STEPS:
                    truncated = True

                total_reward += step_reward
                
                # If the game ends mid-skip, stop the loop immediately
                if terminated or truncated:
                    break
            
            # Return the accumulated reward
            return self._get_obs(), total_reward, terminated, truncated, {}

    def _move_tank(self, tank_id, action):
        body_x, body_y, turret_act, fire_act = action
        
        pos = self.t1_pos if tank_id == 1 else self.t2_pos
        b_angle = self.t1_body_angle if tank_id == 1 else self.t2_body_angle
        t_angle = self.t1_turret_angle if tank_id == 1 else self.t2_turret_angle
        cooldown = self.t1_cooldown if tank_id == 1 else self.t2_cooldown
        
        if cooldown > 0: cooldown -= 1
        
        # Movement
        dx, dy = 0, 0
        if body_x == 1: dx = -self.TANK_SPEED
        elif body_x == 2: dx = self.TANK_SPEED
        
        if body_y == 1: dy = -self.TANK_SPEED
        elif body_y == 2: dy = self.TANK_SPEED
            
        if dx != 0 and dy != 0:
            dx *= 0.707
            dy *= 0.707
            
        pos[0] += dx
        pos[1] += dy
        
        if dx != 0 or dy != 0:
            b_angle = math.degrees(math.atan2(dy, dx)) % 360
            
        # Turret
        if turret_act == 1: t_angle += self.ROT_SPEED_TURRET
        elif turret_act == 2: t_angle -= self.ROT_SPEED_TURRET
        t_angle %= 360
        
        # Fire
        if fire_act == 1 and cooldown == 0:
            self._fire_bullet(tank_id, pos, t_angle)
            cooldown = self.COOLDOWN_MAX
        
        if tank_id == 1:
            self.t1_body_angle = b_angle
            self.t1_turret_angle = t_angle
            self.t1_cooldown = cooldown
        else:
            self.t2_body_angle = b_angle
            self.t2_turret_angle = t_angle
            self.t2_cooldown = cooldown

    def _fire_bullet(self, tank_id, pos, angle):
        rad = math.radians(angle)
        start_pos = pos + np.array([math.cos(rad)*35, math.sin(rad)*35])
        
        gun_jammed = False
        for wall in self.obstacles:
            if wall.collidepoint(start_pos[0], start_pos[1]):
                gun_jammed = True
                break
        
        if gun_jammed:
            return 
            
        vel = np.array([math.cos(rad)*self.BULLET_SPEED, math.sin(rad)*self.BULLET_SPEED])
        
        bullet = self.bullet1 if tank_id == 1 else self.bullet2
        bullet["pos"] = start_pos.copy()
        bullet["vel"] = vel
        bullet["active"] = True

    def _update_bullets(self):
        for b in [self.bullet1, self.bullet2]:
            if b["active"]:
                b["pos"] += b["vel"]
                
                for wall in self.obstacles:
                    if wall.collidepoint(b["pos"][0], b["pos"][1]):
                        b["active"] = False
                        self.visual_effects.append({
                            "pos": b["pos"].copy(),
                            "life": 5,
                            "color": (200, 200, 200)
                        })
                        break
                
                if (b["pos"][0] < 0 or b["pos"][0] > self.grid_size or 
                    b["pos"][1] < 0 or b["pos"][1] > self.grid_size):
                    b["active"] = False

    def _resolve_collisions(self):
        def resolve_circle_rect(circle_pos, circle_radius, rect):
            closest_x = max(rect.left, min(circle_pos[0], rect.right))
            closest_y = max(rect.top, min(circle_pos[1], rect.bottom))
            dx = circle_pos[0] - closest_x
            dy = circle_pos[1] - closest_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < circle_radius and dist > 0:
                overlap = circle_radius - dist
                dx /= dist
                dy /= dist
                return np.array([dx * overlap, dy * overlap])
            return np.zeros(2)

        for pos in [self.t1_pos, self.t2_pos]:
            pos[:] = np.clip(pos, self.TANK_RADIUS, self.grid_size - self.TANK_RADIUS)
            for wall in self.obstacles:
                push = resolve_circle_rect(pos, self.TANK_RADIUS, wall)
                pos += push
            
        dist = np.linalg.norm(self.t1_pos - self.t2_pos)
        min_dist = self.TANK_RADIUS * 2.2
        if dist < min_dist and dist > 0:
            overlap = min_dist - dist
            direction = (self.t1_pos - self.t2_pos) / dist
            self.t1_pos += direction * (overlap / 2)
            self.t2_pos -= direction * (overlap / 2)

    def render(self):
        if self.render_mode is None: return
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.grid_size, self.grid_size))
                pygame.display.set_caption("🎮 Tank Combat AI")
            else:
                self.window = pygame.Surface((self.grid_size, self.grid_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
        canvas = pygame.Surface((self.grid_size, self.grid_size))
        canvas.fill((60, 55, 50))
        
        for x in range(0, self.grid_size, 50):
            pygame.draw.line(canvas, (70, 65, 60), (x, 0), (x, self.grid_size))
            pygame.draw.line(canvas, (70, 65, 60), (0, x), (self.grid_size, x))
            
        for wall in self.obstacles:
            pygame.draw.rect(canvas, (100, 90, 80), wall)
            pygame.draw.rect(canvas, (80, 70, 60), wall, 3)
            pygame.draw.line(canvas, (90, 80, 70), wall.topleft, wall.bottomright)
            pygame.draw.line(canvas, (90, 80, 70), wall.bottomleft, wall.topright)
            
        def draw_tank(pos, b_angle, t_angle, color, label):
            pygame.draw.circle(canvas, (30, 30, 30), (pos + np.array([3, 3])).astype(int), self.TANK_RADIUS)
            pygame.draw.circle(canvas, color, pos.astype(int), self.TANK_RADIUS)
            b_rad = math.radians(b_angle)
            body_end = pos + np.array([math.cos(b_rad)*15, math.sin(b_rad)*15])
            pygame.draw.line(canvas, (30, 30, 30), pos.astype(int), body_end.astype(int), 3)
            t_rad = math.radians(t_angle)
            turret_end = pos + np.array([math.cos(t_rad)*35, math.sin(t_rad)*35])
            pygame.draw.line(canvas, (200, 200, 200), pos.astype(int), turret_end.astype(int), 6)
            if self.font:
                text = self.font.render(label, True, (200, 200, 200))
                canvas.blit(text, (int(pos[0]-10), int(pos[1]-40)))
        
        draw_tank(self.t1_pos, self.t1_body_angle, self.t1_turret_angle, (60, 120, 255), "P1")
        draw_tank(self.t2_pos, self.t2_body_angle, self.t2_turret_angle, (255, 80, 80), "P2")
        
        for b, c in [(self.bullet1, (255, 255, 100)), (self.bullet2, (255, 100, 100))]:
            if b["active"]:
                pygame.draw.circle(canvas, c, b["pos"].astype(int), self.BULLET_RADIUS)
                
        for fx in self.visual_effects[:]:
            if fx["life"] > 0:
                pygame.draw.circle(canvas, fx["color"], fx["pos"].astype(int), 25 - fx["life"])
                fx["life"] -= 1
            else:
                self.visual_effects.remove(fx)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def close(self):
        if self.window: pygame.quit()