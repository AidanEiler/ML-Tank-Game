import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Set to False if you want an open arena without obstacles
USE_WALLS = True  

# Starting difficulty for the bot
# 0 = TURRET (Stationary, good for learning to aim)
# 1 = ZOMBIE (Moves straight at you, good for moving targets)
# 2 = GRUNT  (Avoids walls, chases, uses cover)
# 3 = PRO    (Strafes, leads shots, very aggressive)
DEFAULT_DIFFICULTY = 3
# ==========================================

class HeuristicOpponent:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.tank_speed = 4.0
        self.tank_radius = 20
        self.last_move = (0, 0) 

    def reset_episode(self):
        self.last_move = (0, 0)

    def predict(self, obs, difficulty=3):
        # Indices [0..5]  = PLAYER (Target)
        # Indices [6..11] = BOT (Self)
        # Index [19]      = BOT COOLDOWN
        
        target_x = obs[0] * self.grid_size
        target_y = obs[1] * self.grid_size
        
        my_x = obs[6] * self.grid_size
        my_y = obs[7] * self.grid_size
        
        my_turret_sin = obs[10]
        my_turret_cos = obs[11]
        my_cooldown = obs[19] 

        # Calculate distance and angle to player
        dx = target_x - my_x
        dy = target_y - my_y
        dist_to_player = math.sqrt(dx**2 + dy**2)
        angle_to_player = math.degrees(math.atan2(dy, dx)) % 360
        current_turret_angle = math.degrees(math.atan2(my_turret_sin, my_turret_cos)) % 360
        
        actions = [0, 0, 0, 0] # [BodyX, BodyY, Turret, Fire]

        # --- 1. AIMING LOGIC ---
        aim_error = 30 if difficulty == 0 else (15 if difficulty == 1 else (5 if difficulty == 2 else 0))
        diff = (angle_to_player - current_turret_angle + 180) % 360 - 180

        if abs(diff) > aim_error:
            if diff < 0: actions[2] = 2 
            else: actions[2] = 1        

        # --- 2. FIRING LOGIC ---
        has_los = True
        for wall in self.obstacles:
            # Float cast for safety
            if wall.clipline(float(my_x), float(my_y), float(target_x), float(target_y)):
                has_los = False
                break
        
        if has_los and abs(diff) < 20 and my_cooldown < 0.1:
            fire_prob = [0.05, 0.2, 0.5, 1.0][difficulty]
            if np.random.random() < fire_prob:
                actions[3] = 1

        # --- 3. MOVEMENT LOGIC ---
        if difficulty == 0:
            return np.array(actions), None
            
        move_vec_x, move_vec_y = 0, 0
        
        if difficulty == 1:
            move_vec_x, move_vec_y = dx, dy
        else:
            optimal_dist = 250 if difficulty == 3 else 100
            if dist_to_player < optimal_dist:
                move_vec_x, move_vec_y = -dx, -dy 
            elif dist_to_player > optimal_dist + 50:
                move_vec_x, move_vec_y = dx, dy   
            elif difficulty == 3:
                 move_vec_x, move_vec_y = -dy, dx 
        
        m_len = math.sqrt(move_vec_x**2 + move_vec_y**2)
        if m_len > 0:
            move_vec_x /= m_len
            move_vec_y /= m_len

        # --- 4. COLLISION AVOIDANCE ---
        best_score = -9999
        best_move = (0, 0)
        
        moves = [
            (0,0, 0,0), (1,0, -1,0), (2,0, 1,0), (0,1, 0,-1), (0,2, 0,1),
            (1,1, -0.7,-0.7), (2,1, 0.7,-0.7), (1,2, -0.7,0.7), (2,2, 0.7,0.7)
        ]
        
        for bx, by, v_x, v_y in moves:
            # Base Score
            score = (v_x * move_vec_x) + (v_y * move_vec_y) * 2.0 
            
            # Only give Inertia Bonus if we are MOVING.
            # We do NOT want to reward standing still (0,0).
            if (bx, by) == self.last_move and (bx, by) != (0, 0):
                score += 2.0 

            # Tiny noise to break ties
            score += np.random.uniform(0, 0.1)

            # Look ahead
            future_x = my_x + (v_x * self.tank_speed * 5)
            future_y = my_y + (v_y * self.tank_speed * 5)
            
            # Penalty for Map Borders
            if future_x < 20 or future_x > self.grid_size - 20 or future_y < 20 or future_y > self.grid_size - 20:
                score -= 10.0 
            
            # Penalty for Walls
            player_rect = pygame.Rect(float(future_x - 15), float(future_y - 15), 30, 30)
            if player_rect.collidelist(self.obstacles) != -1:
                score -= 10.0 
            
            if score > best_score:
                best_score = score
                best_move = (bx, by)
        
        self.last_move = best_move
        
        actions[0], actions[1] = best_move
        return np.array(actions), None

class TankCombatEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, grid_size=600, opponent_type="bot", opponent_policy=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.opponent_policy = opponent_policy
        self.bot_difficulty = DEFAULT_DIFFICULTY 
        
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 2])
        
        # === OBSERVATION SPACE ===
        # 20 Basic Inputs + 8 Lidar Sensors = 28 Total Inputs
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(28,), dtype=np.float32
        )
        
        # Game Constants
        self.TANK_SPEED = 4.0
        self.ROT_SPEED_TURRET = 5.0 
        self.BULLET_SPEED = 12.0
        self.TANK_RADIUS = 20
        self.BULLET_RADIUS = 5
        self.MAX_STEPS = 3000
        self.COOLDOWN_MAX = 40
        self.LIDAR_RANGE = 250.0 # Range of the AI's vision whiskers
        
        self.last_t1_pos = None
        self.last_t2_pos = None
        
        # Generate Obstacles
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

        self.heuristic_bot = HeuristicOpponent(self.grid_size, self.obstacles)
        self.window = None
        self.clock = None
        self.font = None

    def _get_random_spawn(self, y_min, y_max, avoid_pos=None):
        padding = self.TANK_RADIUS + 5
        min_dist = 100 
        for _ in range(100): 
            x = np.random.uniform(padding, self.grid_size - padding)
            y = np.random.uniform(y_min + padding, y_max - padding)
            candidate = np.array([x, y], dtype=np.float32)
            
            # Don't spawn too close to where someone just died
            if avoid_pos is not None and np.linalg.norm(candidate - avoid_pos) < min_dist:
                continue 
            
            # Don't spawn inside a wall
            # Explicit float cast for Rect
            temp_rect = pygame.Rect(float(x - 20), float(y - 20), 40, 40)
            if temp_rect.collidelist(self.obstacles) == -1:
                return candidate
                
        # Fallback to center if random generation fails
        return np.array([self.grid_size/2, (y_min + y_max)/2], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.opponent_type == "bot": self.heuristic_bot.reset_episode()
        
        quarter_map = self.grid_size * 0.25
        
        # Reset Player
        self.t1_pos = self._get_random_spawn(0, quarter_map, avoid_pos=self.last_t1_pos)
        self.t1_body_angle = 90 
        self.t1_turret_angle = 90
        self.t1_cooldown = 0
        
        # Reset Enemy
        self.t2_pos = self._get_random_spawn(self.grid_size - quarter_map, self.grid_size, avoid_pos=self.last_t2_pos)
        self.t2_body_angle = 270 
        self.t2_turret_angle = 270
        self.t2_cooldown = 0
        
        self.last_t1_pos = self.t1_pos.copy()
        self.last_t2_pos = self.t2_pos.copy()
        
        self.bullet1 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        self.bullet2 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        self.step_count = 0
        self.visual_effects = []
        
        return self._get_obs(), {}

    # --- SENSORS ---
    def _get_lidar(self, pos, body_angle):
        """
        Casts 8 rays from the tank center to detect walls.
        Returns values 0.0 (Wall is touching me) to 1.0 (Clear space).
        """
        readings = []
        # 8 directions: 0, 45, 90, 135, 180, 225, 270, 315
        for i in range(8):
            rel_angle = i * 45
            abs_angle = math.radians((body_angle + rel_angle) % 360)
            
            start_pos = pos
            end_pos = pos + np.array([math.cos(abs_angle), math.sin(abs_angle)]) * self.LIDAR_RANGE
            
            closest_dist = self.LIDAR_RANGE
            
            # 1. Check Map Borders
            if end_pos[0] < 0: closest_dist = min(closest_dist, pos[0])
            if end_pos[0] > self.grid_size: closest_dist = min(closest_dist, self.grid_size - pos[0])
            if end_pos[1] < 0: closest_dist = min(closest_dist, pos[1])
            if end_pos[1] > self.grid_size: closest_dist = min(closest_dist, self.grid_size - pos[1])
            
            # 2. Check Obstacles
            for wall in self.obstacles:
                # Explicit float cast to prevent numpy crash
                clipped = wall.clipline(float(start_pos[0]), float(start_pos[1]), float(end_pos[0]), float(end_pos[1]))
                if clipped:
                    d = math.hypot(clipped[0][0] - start_pos[0], clipped[0][1] - start_pos[1])
                    if d < closest_dist:
                        closest_dist = d
            
            readings.append(closest_dist / self.LIDAR_RANGE)
            
        return np.array(readings, dtype=np.float32)

    def _get_obs(self):
        def norm_pos(val): return val / self.grid_size
        t1_b_rad = math.radians(self.t1_body_angle)
        t1_t_rad = math.radians(self.t1_turret_angle)
        t2_b_rad = math.radians(self.t2_body_angle)
        t2_t_rad = math.radians(self.t2_turret_angle)
        
        # Get Lidar readings for Player
        lidar_readings = self._get_lidar(self.t1_pos, self.t1_body_angle)
        
        base_obs = np.array([
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
        
        # Combine Basic Stats + Lidar
        return np.concatenate([base_obs, lidar_readings])
    
    def _get_opponent_obs(self):
        # Used for self-play training
        def norm_pos(val): return val / self.grid_size
        t1_b_rad = math.radians(self.t1_body_angle)
        t1_t_rad = math.radians(self.t1_turret_angle)
        t2_b_rad = math.radians(self.t2_body_angle)
        t2_t_rad = math.radians(self.t2_turret_angle)
        
        lidar_readings = self._get_lidar(self.t2_pos, self.t2_body_angle)
        
        base_obs = np.array([
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
        
        return np.concatenate([base_obs, lidar_readings])

    def step(self, action):
            total_reward = 0
            terminated = False
            truncated = False
            
            # Frame Skip: 1 for smooth viewing, 4 for fast training
            if self.render_mode in ["human", "rgb_array"]:
                repeat_frames = 1 
            else:
                repeat_frames = 4 
            
            for _ in range(repeat_frames):
                if self.render_mode == "human": self.render()
                self.step_count += 1
                
                # 1. AI Logic
                if self.opponent_type == "bot":
                    op_obs = self._get_obs() 
                    op_action, _ = self.heuristic_bot.predict(op_obs, difficulty=self.bot_difficulty)
                elif self.opponent_type in ["self", "policy"] and self.opponent_policy:
                    op_obs = self._get_opponent_obs()
                    op_action, _ = self.opponent_policy.predict(op_obs, deterministic=False)
                else:
                    op_action = self.action_space.sample()
                
                # 2. Physics
                self._move_tank(1, action)
                self._move_tank(2, op_action)
                self._resolve_collisions()
                self._update_bullets()
                
                # 3. Rewards
                step_reward = -0.005 # Time penalty
                
                # Check hits
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

                if self.step_count >= self.MAX_STEPS: truncated = True
                total_reward += step_reward
                
                if terminated or truncated: break
            
            return self._get_obs(), total_reward, terminated, truncated, {}

    def _move_tank(self, tank_id, action):
        body_x, body_y, turret_act, fire_act = action
        pos = self.t1_pos if tank_id == 1 else self.t2_pos
        b_angle = self.t1_body_angle if tank_id == 1 else self.t2_body_angle
        t_angle = self.t1_turret_angle if tank_id == 1 else self.t2_turret_angle
        cooldown = self.t1_cooldown if tank_id == 1 else self.t2_cooldown
        
        if cooldown > 0: cooldown -= 1
        
        # Parse Movement (0=Stop, 1=Reverse, 2=Forward)
        dx, dy = 0, 0
        if body_x == 1: dx = -self.TANK_SPEED
        elif body_x == 2: dx = self.TANK_SPEED
        if body_y == 1: dy = -self.TANK_SPEED
        elif body_y == 2: dy = self.TANK_SPEED
            
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            dx *= 0.707
            dy *= 0.707
            
        pos[0] += dx
        pos[1] += dy
        
        # Rotate body to face movement direction
        if dx != 0 or dy != 0:
            b_angle = math.degrees(math.atan2(dy, dx)) % 360
            
        # Parse Turret Rotation
        if turret_act == 1: t_angle += self.ROT_SPEED_TURRET
        elif turret_act == 2: t_angle -= self.ROT_SPEED_TURRET
        t_angle %= 360
        
        # Fire Bullet
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
        
        # Check if gun is inside a wall (anti-clipping)
        gun_jammed = False
        for wall in self.obstacles:
            if wall.collidepoint(start_pos[0], start_pos[1]):
                gun_jammed = True
                break
        if gun_jammed: return 
            
        vel = np.array([math.cos(rad)*self.BULLET_SPEED, math.sin(rad)*self.BULLET_SPEED])
        bullet = self.bullet1 if tank_id == 1 else self.bullet2
        bullet["pos"] = start_pos.copy()
        bullet["vel"] = vel
        bullet["active"] = True

    def _update_bullets(self):
        for b in [self.bullet1, self.bullet2]:
            if b["active"]:
                b["pos"] += b["vel"]
                # Check Wall Hit
                for wall in self.obstacles:
                    if wall.collidepoint(b["pos"][0], b["pos"][1]):
                        b["active"] = False
                        self.visual_effects.append({"pos": b["pos"].copy(), "life": 5, "color": (200, 200, 200)})
                        break
                # Check Map Boundary
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
            
        # Tank vs Tank Collision
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
                pygame.display.set_caption("🎮 Tank Combat AI (Lidar Enabled)")
            else:
                self.window = pygame.Surface((self.grid_size, self.grid_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
        canvas = pygame.Surface((self.grid_size, self.grid_size))
        canvas.fill((60, 55, 50))
        
        # Draw Grid
        for x in range(0, self.grid_size, 50):
            pygame.draw.line(canvas, (70, 65, 60), (x, 0), (x, self.grid_size))
            pygame.draw.line(canvas, (70, 65, 60), (0, x), (self.grid_size, x))
            
        # Draw Walls
        for wall in self.obstacles:
            pygame.draw.rect(canvas, (100, 90, 80), wall)
            pygame.draw.rect(canvas, (80, 70, 60), wall, 3)
            pygame.draw.line(canvas, (90, 80, 70), wall.topleft, wall.bottomright)
            pygame.draw.line(canvas, (90, 80, 70), wall.bottomleft, wall.topright)
            
        # Draw Lidar Lines (Visual Debugging for Player 1)
        # These green lines show what the AI "sees"
        for i in range(8):
            rel_angle = i * 45
            abs_angle = math.radians((self.t1_body_angle + rel_angle) % 360)
            start = self.t1_pos
            end = start + np.array([math.cos(abs_angle), math.sin(abs_angle)]) * self.LIDAR_RANGE
            
            # Re-calculate hit point for drawing
            closest_dist = self.LIDAR_RANGE
            
            # Map Borders
            if end[0] < 0: closest_dist = min(closest_dist, start[0])
            if end[0] > self.grid_size: closest_dist = min(closest_dist, self.grid_size - start[0])
            if end[1] < 0: closest_dist = min(closest_dist, start[1])
            if end[1] > self.grid_size: closest_dist = min(closest_dist, self.grid_size - start[1])

            # Obstacles
            for wall in self.obstacles:
                # Explicit float cast for RENDER logic
                clipped = wall.clipline(float(start[0]), float(start[1]), float(end[0]), float(end[1]))
                if clipped:
                    d = math.hypot(clipped[0][0] - start[0], clipped[0][1] - start[1])
                    if d < closest_dist: closest_dist = d
            
            final_end = start + np.array([math.cos(abs_angle), math.sin(abs_angle)]) * closest_dist
            
            # Convert start/end to Tuple(float, float) for Pygame Line Drawing
            start_tuple = (float(start[0]), float(start[1]))
            end_tuple = (float(final_end[0]), float(final_end[1]))
            
            pygame.draw.line(canvas, (50, 255, 50), start_tuple, end_tuple, 1)
            pygame.draw.circle(canvas, (50, 255, 50), (int(final_end[0]), int(final_end[1])), 3)

        def draw_tank(pos, b_angle, t_angle, color, label):
            # Body
            pygame.draw.circle(canvas, (30, 30, 30), (pos + np.array([3, 3])).astype(int), self.TANK_RADIUS)
            pygame.draw.circle(canvas, color, pos.astype(int), self.TANK_RADIUS)
            # Direction Indicator
            b_rad = math.radians(b_angle)
            body_end = pos + np.array([math.cos(b_rad)*15, math.sin(b_rad)*15])
            pygame.draw.line(canvas, (30, 30, 30), pos.astype(int), body_end.astype(int), 3)
            # Turret
            t_rad = math.radians(t_angle)
            turret_end = pos + np.array([math.cos(t_rad)*35, math.sin(t_rad)*35])
            pygame.draw.line(canvas, (200, 200, 200), pos.astype(int), turret_end.astype(int), 6)
            # Label
            if self.font:
                text = self.font.render(label, True, (200, 200, 200))
                canvas.blit(text, (int(pos[0]-10), int(pos[1]-40)))
        
        draw_tank(self.t1_pos, self.t1_body_angle, self.t1_turret_angle, (60, 120, 255), "P1")
        draw_tank(self.t2_pos, self.t2_body_angle, self.t2_turret_angle, (255, 80, 80), "P2")
        
        # Bullets
        for b, c in [(self.bullet1, (255, 255, 100)), (self.bullet2, (255, 100, 100))]:
            if b["active"]:
                pygame.draw.circle(canvas, c, b["pos"].astype(int), self.BULLET_RADIUS)
        
        # Visual Effects (Explosions)
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