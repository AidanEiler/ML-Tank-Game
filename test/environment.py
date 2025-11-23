"""
environment.py - tank combat gymnasium environment

a complete reinforcement learning environment for training ai agents
to play a 2d tank combat game. includes:
    - the tank combat game (TankCombatEnv)
    - a heuristic bot opponent with 4 difficulty levels (HeuristicOpponent)
    - factory function for easy environment creation (create_environment)

based on gymnasium (openai gym fork) and pygame for rendering.
"""

import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# heuristic opponent - rule-based bot with 4 difficulty levels
# =============================================================================

class HeuristicOpponent:
    """
    rule-based opponent for the tank combat environment.
    
    difficulty levels:
        0 = zombie: slow, stops when close, bad aim, rarely shoots
        1 = noob: normal speed, gets stuck on walls, bad aim
        2 = average: decent aim, standard movement, slight delay
        3 = pro: perfect aim, slides around corners, instant fire
    """
    
    def __init__(self, difficulty=0, use_walls=True, grid_size=600):
        """
        initialize the heuristic opponent.
        
        args:
            difficulty: int (0-3), how smart the bot is
            use_walls: bool, whether walls exist in the arena
            grid_size: int, size of the arena in pixels
        """
        self.difficulty = difficulty
        self.grid_size = grid_size
        self.move_timer = 0
        self.current_action = [0, 0, 0, 0]
        
        # randomized behavior thresholds
        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)

        # build obstacle list
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
        """reset per-episode state at the start of each match."""
        self.move_timer = 0
        self.current_action = [0, 0, 0, 0]
        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)

    def predict(self, obs, deterministic=True):
        """
        decide what action to take based on observation.
        
        args:
            obs: numpy array of observations from the environment
            deterministic: unused, kept for api compatibility
            
        returns:
            tuple: (action array, None)
        """
        # random chaos for lower difficulty bots
        chaos_thresh = 0.10 if self.difficulty <= 1 else 0.05
        if np.random.random() < chaos_thresh:
            rand_act = np.array([
                np.random.randint(0, 3), 
                np.random.randint(0, 3), 
                np.random.randint(0, 3), 
                np.random.randint(0, 2)
            ])
            return rand_act, None

        # extract positions from observation
        # note: obs indices are from the bot's perspective (swapped in _get_opponent_obs)
        norm_me_x, norm_me_y = obs[6], obs[7]
        norm_enemy_x, norm_enemy_y = obs[0], obs[1]
        
        me_turret_sin, me_turret_cos = obs[10], obs[11]
        me_turret_angle = math.degrees(math.atan2(me_turret_sin, me_turret_cos)) % 360
        
        cooldown = obs[19] 
        
        # calculate distance and angle to enemy
        dx = norm_enemy_x - norm_me_x
        dy = norm_enemy_y - norm_me_y
        dist = math.sqrt(dx**2 + dy**2)
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        turret_diff = (target_angle - me_turret_angle + 180) % 360 - 180
        
        # convert to pixel positions
        pix_me_x = float(norm_me_x * self.grid_size)
        pix_me_y = float(norm_me_y * self.grid_size)
        pix_enemy_x = float(norm_enemy_x * self.grid_size)
        pix_enemy_y = float(norm_enemy_y * self.grid_size)

        # check line of sight
        has_los = True
        for wall in self.obstacles:
            if wall.clipline(pix_me_x, pix_me_y, pix_enemy_x, pix_enemy_y):
                has_los = False
                break

        # check if barrel is blocked by a wall
        barrel_rad = math.radians(me_turret_angle)
        barrel_x = pix_me_x + math.cos(barrel_rad) * 35
        barrel_y = pix_me_y + math.sin(barrel_rad) * 35
        
        barrel_blocked = False
        for wall in self.obstacles:
            if wall.collidepoint(barrel_x, barrel_y):
                barrel_blocked = True
                break
        
        # initialize actions: [body_x, body_y, turret, fire]
        actions = [0, 0, 0, 0]

        # aiming logic - lower difficulty = worse aim
        aim_margin = {0: 40, 1: 20, 2: 10, 3: 5}.get(self.difficulty, 5)
        
        if abs(turret_diff) > aim_margin:
            actions[2] = 2 if turret_diff < 0 else 1
            
        # firing logic - lower difficulty = less likely to fire
        fire_prob = {0: 0.01, 1: 0.2, 2: 0.5, 3: 1.0}.get(self.difficulty, 1.0)
        
        if has_los and not barrel_blocked and abs(turret_diff) < 15 and cooldown < 0.1:
            if np.random.random() < fire_prob:
                actions[3] = 1 

        # movement logic
        def is_move_safe(ax, ay):
            """check if a movement would collide with walls or boundaries."""
            vx, vy = 0, 0
            if ax == 1: vx = -40 
            elif ax == 2: vx = 40 
            if ay == 1: vy = -40 
            elif ay == 2: vy = 40
            
            future_x = pix_me_x + vx
            future_y = pix_me_y + vy
            
            if future_x < 20 or future_x > self.grid_size - 20:
                return False
            if future_y < 20 or future_y > self.grid_size - 20:
                return False
            
            future_rect = pygame.Rect(future_x - 20, future_y - 20, 40, 40)
            if future_rect.collidelist(self.obstacles) != -1:
                return False
            return True

        # check if current movement plan is still valid
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

        # pick new movement plan
        if self.difficulty == 0:
            self.move_timer = 5
        elif self.difficulty == 1:
            self.move_timer = np.random.randint(30, 60)
        else:
            self.move_timer = np.random.randint(5, 25)

        self.retreat_dist = np.random.uniform(0.2, 0.4)
        self.chase_dist = np.random.uniform(0.5, 0.7)
        
        ideal_ax, ideal_ay = 0, 0
        
        # zombie logic (difficulty 0): slow walk toward enemy
        if self.difficulty == 0:
            ideal_ax = 2 if dx > 0 else 1
            ideal_ay = 2 if dy > 0 else 1
            
            # stop if close enough
            if dist < 0.4:
                ideal_ax, ideal_ay = 0, 0
            # stutter step (70% chance to stand still)
            elif np.random.random() > 0.3:
                ideal_ax, ideal_ay = 0, 0
        
        # smarter logic (difficulty 1-3)
        else:
            if not has_los:
                # no line of sight, try to get around obstacle
                if abs(dx) > abs(dy): 
                    ideal_ay = 2 if dy > 0 else 1
                else:
                    ideal_ax = 2 if dx > 0 else 1
            else:
                if dist < self.retreat_dist: 
                    # too close, retreat
                    ideal_ax = 1 if dx > 0 else 2
                    ideal_ay = 1 if dy > 0 else 2
                elif dist > self.chase_dist: 
                    # too far, chase
                    ideal_ax = 2 if dx > 0.05 else (1 if dx < -0.05 else 0)
                    ideal_ay = 2 if dy > 0.05 else (1 if dy < -0.05 else 0)
                else: 
                    # good distance, strafe perpendicular
                    p_dx, p_dy = -dy, dx
                    ideal_ax = 2 if p_dx > 0 else 1
                    ideal_ay = 2 if p_dy > 0 else 1

        # collision resolution
        if is_move_safe(ideal_ax, ideal_ay):
            actions[0], actions[1] = ideal_ax, ideal_ay
        else:
            # only pro (difficulty 3) knows how to slide around corners
            if self.difficulty == 3:
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
                # lower difficulty bots just stop when blocked
                actions[0], actions[1] = 0, 0

        self.current_action = actions
        return np.array(actions), None


# =============================================================================
# tank combat environment - the actual game
# =============================================================================

class TankCombatEnv(gym.Env):
    """
    a 2d tank combat environment for reinforcement learning.
    
    two tanks fight in an arena. the player (tank 1) tries to shoot
    the opponent (tank 2) while avoiding getting shot.
    
    action space (MultiDiscrete([3, 3, 3, 2])):
        [0] body_x: 0=none, 1=left, 2=right
        [1] body_y: 0=none, 1=up, 2=down
        [2] turret: 0=none, 1=clockwise, 2=counter-clockwise
        [3] fire: 0=no, 1=yes
    
    observation space (Box, 20 values):
        positions, angles (as sin/cos), bullet states, cooldowns
        all normalized to [-1, 1] range
    
    rewards:
        +5.0 for hitting the enemy
        -5.0 for getting hit
        -0.005 per step (encourages quick wins)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        bot_difficulty=0,
        use_walls=False,
        render_mode=None,
        grid_size=600,
        max_steps=3000,
        opponent_type="bot",
        opponent_policy=None
    ):
        """
        initialize the tank combat environment.
        
        args:
            bot_difficulty: int (0-3), difficulty of the heuristic opponent
            use_walls: bool, whether to include obstacle walls
            render_mode: None, "human", or "rgb_array"
            grid_size: int, arena size in pixels
            max_steps: int, maximum steps before timeout
            opponent_type: "bot" for heuristic, "self" or "policy" for trained model
            opponent_policy: policy object if opponent_type is "self" or "policy"
        """
        super().__init__()
        
        self.bot_difficulty = bot_difficulty
        self.use_walls = use_walls
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.opponent_type = opponent_type
        self.opponent_policy = opponent_policy
        
        # action space: [body_x, body_y, turret, fire]
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 2])
        
        # observation space: 20 normalized values
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # game constants
        self.TANK_SPEED = 4.0
        self.ROT_SPEED_TURRET = 5.0 
        self.BULLET_SPEED = 12.0
        self.TANK_RADIUS = 20
        self.BULLET_RADIUS = 5
        self.MAX_STEPS = max_steps
        self.COOLDOWN_MAX = 40
        
        # spawn tracking (to avoid spawning in same spot twice)
        self.last_t1_pos = None
        self.last_t2_pos = None
        
        # create heuristic bot opponent
        self.heuristic_bot = HeuristicOpponent(
            difficulty=bot_difficulty,
            use_walls=use_walls,
            grid_size=grid_size
        )
        
        # build obstacle list
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
        
        # pygame state (initialized lazily)
        self.window = None
        self.clock = None
        self.font = None
        
        # visual effects (explosions, etc)
        self.visual_effects = []

    def _get_random_spawn(self, y_min, y_max, avoid_pos=None):
        """
        find a spawn point that doesn't collide with walls
        and is away from the previous spawn location.
        """
        padding = self.TANK_RADIUS + 5
        min_dist_from_last = 100
        
        for _ in range(100): 
            x = np.random.uniform(padding, self.grid_size - padding)
            y = np.random.uniform(y_min + padding, y_max - padding)
            candidate = np.array([x, y], dtype=np.float32)
            
            # check distance from last spawn
            if avoid_pos is not None:
                dist = np.linalg.norm(candidate - avoid_pos)
                if dist < min_dist_from_last:
                    continue
            
            # check wall collision
            temp_rect = pygame.Rect(
                x - self.TANK_RADIUS, 
                y - self.TANK_RADIUS, 
                self.TANK_RADIUS * 2, 
                self.TANK_RADIUS * 2
            )
            
            if temp_rect.collidelist(self.obstacles) == -1:
                return candidate
        
        # fallback to center of zone
        return np.array([self.grid_size/2, (y_min + y_max)/2], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """reset the environment for a new episode."""
        super().reset(seed=seed)
        
        if self.opponent_type == "bot":
            self.heuristic_bot.reset_episode()
        
        quarter_map = self.grid_size * 0.25
        
        # spawn tank 1 (player) in top quarter
        self.t1_pos = self._get_random_spawn(0, quarter_map, avoid_pos=self.last_t1_pos)
        self.t1_body_angle = 90 
        self.t1_turret_angle = 90
        self.t1_cooldown = 0
        
        # spawn tank 2 (enemy) in bottom quarter
        self.t2_pos = self._get_random_spawn(
            self.grid_size - quarter_map, 
            self.grid_size, 
            avoid_pos=self.last_t2_pos
        )
        self.t2_body_angle = 270 
        self.t2_turret_angle = 270
        self.t2_cooldown = 0
        
        # save spawn positions for next reset
        self.last_t1_pos = self.t1_pos.copy()
        self.last_t2_pos = self.t2_pos.copy()
        
        # initialize bullets
        self.bullet1 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        self.bullet2 = {"pos": np.zeros(2), "vel": np.zeros(2), "active": False}
        
        self.step_count = 0
        self.visual_effects = []
        
        return self._get_obs(), {}

    def _get_obs(self):
        """get observation from player's perspective."""
        def norm_pos(val): 
            return val / self.grid_size
            
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
        """get observation from opponent's perspective (positions swapped)."""
        def norm_pos(val): 
            return val / self.grid_size
            
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
        """execute one step in the environment."""
        total_reward = 0
        terminated = False
        truncated = False
        
        # frame skip: faster training, but smooth rendering for humans
        if self.render_mode in ["human", "rgb_array"]:
            repeat_frames = 1
        else:
            repeat_frames = 4
        
        for _ in range(repeat_frames):
            if self.render_mode == "human":
                self.render()
                
            self.step_count += 1
            
            # get opponent action
            if self.opponent_type == "bot":
                op_obs = self._get_obs()
                op_action, _ = self.heuristic_bot.predict(op_obs)
            elif self.opponent_type in ["self", "policy"] and self.opponent_policy:
                op_obs = self._get_opponent_obs()
                op_action, _ = self.opponent_policy.predict(op_obs, deterministic=False)
            else:
                op_action = self.action_space.sample()
            
            # move tanks
            self._move_tank(1, action)
            self._move_tank(2, op_action)
            
            # resolve collisions
            self._resolve_collisions()
            self._update_bullets()
            
            # small time penalty to encourage quick wins
            step_reward = -0.005
            
            # check if player hit enemy
            if self.bullet1["active"]:
                dist = np.linalg.norm(self.bullet1["pos"] - self.t2_pos)
                if dist < self.TANK_RADIUS + self.BULLET_RADIUS:
                    step_reward += 5.0
                    terminated = True
                    self.bullet1["active"] = False 
                    self.visual_effects.append({
                        "pos": self.t2_pos.copy(), 
                        "life": 15, 
                        "color": (255, 50, 50)
                    })
            
            # check if enemy hit player
            if self.bullet2["active"]:
                dist = np.linalg.norm(self.bullet2["pos"] - self.t1_pos)
                if dist < self.TANK_RADIUS + self.BULLET_RADIUS:
                    step_reward -= 5.0
                    terminated = True
                    self.bullet2["active"] = False 
                    self.visual_effects.append({
                        "pos": self.t1_pos.copy(), 
                        "life": 15, 
                        "color": (50, 150, 255)
                    })

            # check timeout
            if self.step_count >= self.MAX_STEPS:
                truncated = True

            total_reward += step_reward
            
            if terminated or truncated:
                break
        
        return self._get_obs(), total_reward, terminated, truncated, {}

    def _move_tank(self, tank_id, action):
        """move a tank based on action."""
        body_x, body_y, turret_act, fire_act = action
        
        if tank_id == 1:
            pos = self.t1_pos
            b_angle = self.t1_body_angle
            t_angle = self.t1_turret_angle
            cooldown = self.t1_cooldown
        else:
            pos = self.t2_pos
            b_angle = self.t2_body_angle
            t_angle = self.t2_turret_angle
            cooldown = self.t2_cooldown
        
        if cooldown > 0:
            cooldown -= 1
        
        # movement
        dx, dy = 0, 0
        if body_x == 1:
            dx = -self.TANK_SPEED
        elif body_x == 2:
            dx = self.TANK_SPEED
        
        if body_y == 1:
            dy = -self.TANK_SPEED
        elif body_y == 2:
            dy = self.TANK_SPEED
            
        # normalize diagonal movement
        if dx != 0 and dy != 0:
            dx *= 0.707
            dy *= 0.707
            
        pos[0] += dx
        pos[1] += dy
        
        # update body angle to face movement direction
        if dx != 0 or dy != 0:
            b_angle = math.degrees(math.atan2(dy, dx)) % 360
            
        # turret rotation
        if turret_act == 1:
            t_angle += self.ROT_SPEED_TURRET
        elif turret_act == 2:
            t_angle -= self.ROT_SPEED_TURRET
        t_angle %= 360
        
        # fire
        if fire_act == 1 and cooldown == 0:
            self._fire_bullet(tank_id, pos, t_angle)
            cooldown = self.COOLDOWN_MAX
        
        # save state
        if tank_id == 1:
            self.t1_body_angle = b_angle
            self.t1_turret_angle = t_angle
            self.t1_cooldown = cooldown
        else:
            self.t2_body_angle = b_angle
            self.t2_turret_angle = t_angle
            self.t2_cooldown = cooldown

    def _fire_bullet(self, tank_id, pos, angle):
        """fire a bullet from a tank's turret."""
        rad = math.radians(angle)
        start_pos = pos + np.array([math.cos(rad)*35, math.sin(rad)*35])
        
        # check if barrel is inside a wall
        gun_jammed = False
        for wall in self.obstacles:
            if wall.collidepoint(start_pos[0], start_pos[1]):
                gun_jammed = True
                break
        
        if gun_jammed:
            return 
            
        vel = np.array([
            math.cos(rad) * self.BULLET_SPEED, 
            math.sin(rad) * self.BULLET_SPEED
        ])
        
        bullet = self.bullet1 if tank_id == 1 else self.bullet2
        bullet["pos"] = start_pos.copy()
        bullet["vel"] = vel
        bullet["active"] = True

    def _update_bullets(self):
        """update bullet positions and check for wall collisions."""
        for b in [self.bullet1, self.bullet2]:
            if b["active"]:
                b["pos"] += b["vel"]
                
                # wall collision
                for wall in self.obstacles:
                    if wall.collidepoint(b["pos"][0], b["pos"][1]):
                        b["active"] = False
                        self.visual_effects.append({
                            "pos": b["pos"].copy(),
                            "life": 5,
                            "color": (200, 200, 200)
                        })
                        break
                
                # boundary collision
                if (b["pos"][0] < 0 or b["pos"][0] > self.grid_size or 
                    b["pos"][1] < 0 or b["pos"][1] > self.grid_size):
                    b["active"] = False

    def _resolve_collisions(self):
        """resolve tank-wall and tank-tank collisions."""
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

        # resolve each tank against walls and boundaries
        for pos in [self.t1_pos, self.t2_pos]:
            pos[:] = np.clip(pos, self.TANK_RADIUS, self.grid_size - self.TANK_RADIUS)
            for wall in self.obstacles:
                push = resolve_circle_rect(pos, self.TANK_RADIUS, wall)
                pos += push
        
        # resolve tank-tank collision
        dist = np.linalg.norm(self.t1_pos - self.t2_pos)
        min_dist = self.TANK_RADIUS * 2.2
        if dist < min_dist and dist > 0:
            overlap = min_dist - dist
            direction = (self.t1_pos - self.t2_pos) / dist
            self.t1_pos += direction * (overlap / 2)
            self.t2_pos -= direction * (overlap / 2)

    def render(self):
        """render the game to screen or return pixel array."""
        if self.render_mode is None:
            return
            
        # initialize pygame on first render
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode((self.grid_size, self.grid_size))
                pygame.display.set_caption("tank combat ai")
            else:
                self.window = pygame.Surface((self.grid_size, self.grid_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
        canvas = pygame.Surface((self.grid_size, self.grid_size))
        canvas.fill((60, 55, 50))
        
        # draw grid
        for x in range(0, self.grid_size, 50):
            pygame.draw.line(canvas, (70, 65, 60), (x, 0), (x, self.grid_size))
            pygame.draw.line(canvas, (70, 65, 60), (0, x), (self.grid_size, x))
        
        # draw walls
        for wall in self.obstacles:
            pygame.draw.rect(canvas, (100, 90, 80), wall)
            pygame.draw.rect(canvas, (80, 70, 60), wall, 3)
            pygame.draw.line(canvas, (90, 80, 70), wall.topleft, wall.bottomright)
            pygame.draw.line(canvas, (90, 80, 70), wall.bottomleft, wall.topright)
            
        # draw tanks
        def draw_tank(pos, b_angle, t_angle, color, label):
            # shadow
            pygame.draw.circle(
                canvas, (30, 30, 30), 
                (pos + np.array([3, 3])).astype(int), 
                self.TANK_RADIUS
            )
            # body
            pygame.draw.circle(canvas, color, pos.astype(int), self.TANK_RADIUS)
            # body direction indicator
            b_rad = math.radians(b_angle)
            body_end = pos + np.array([math.cos(b_rad)*15, math.sin(b_rad)*15])
            pygame.draw.line(canvas, (30, 30, 30), pos.astype(int), body_end.astype(int), 3)
            # turret
            t_rad = math.radians(t_angle)
            turret_end = pos + np.array([math.cos(t_rad)*35, math.sin(t_rad)*35])
            pygame.draw.line(canvas, (200, 200, 200), pos.astype(int), turret_end.astype(int), 6)
            # label
            if self.font:
                text = self.font.render(label, True, (200, 200, 200))
                canvas.blit(text, (int(pos[0]-10), int(pos[1]-40)))
        
        draw_tank(self.t1_pos, self.t1_body_angle, self.t1_turret_angle, (60, 120, 255), "P1")
        draw_tank(self.t2_pos, self.t2_body_angle, self.t2_turret_angle, (255, 80, 80), "P2")
        
        # draw bullets
        for b, c in [(self.bullet1, (255, 255, 100)), (self.bullet2, (255, 100, 100))]:
            if b["active"]:
                pygame.draw.circle(canvas, c, b["pos"].astype(int), self.BULLET_RADIUS)
                
        # draw visual effects
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
        """clean up pygame resources."""
        if self.window:
            pygame.quit()


# =============================================================================
# factory function and utilities
# =============================================================================

def create_environment(
    bot_difficulty=0,
    use_walls=False,
    render_mode=None,
    grid_size=600,
    max_steps=3000
):
    """
    create and return a configured tank combat environment.
    
    this is the recommended way to create environments, as it validates
    parameters and provides a clean interface.
    
    args:
        bot_difficulty: int (0-3)
            0 = zombie (slow, stops when close, bad aim)
            1 = noob (normal speed, gets stuck on walls, bad aim)
            2 = average (decent aim, standard movement, slight delay)
            3 = pro (perfect aim, slides around corners, instant fire)
        use_walls: bool
            whether to include obstacle walls in the arena
        render_mode: str or None
            none = no rendering (fastest, for training)
            "human" = render to screen (for watching)
            "rgb_array" = return pixel data (for recording)
        grid_size: int
            size of the arena in pixels (default 600x600)
        max_steps: int
            maximum steps before episode is truncated (timeout)
    
    returns:
        configured TankCombatEnv instance
    """
    
    # validate bot difficulty
    if bot_difficulty not in [0, 1, 2, 3]:
        raise ValueError(f"bot_difficulty must be 0-3, got {bot_difficulty}")
    
    # validate render mode
    valid_render_modes = [None, "human", "rgb_array"]
    if render_mode not in valid_render_modes:
        raise ValueError(f"render_mode must be one of {valid_render_modes}, got {render_mode}")
    
    # validate grid size
    if grid_size < 200 or grid_size > 2000:
        raise ValueError(f"grid_size must be between 200-2000, got {grid_size}")
    
    # validate max steps
    if max_steps < 100:
        raise ValueError(f"max_steps must be at least 100, got {max_steps}")
    
    return TankCombatEnv(
        bot_difficulty=bot_difficulty,
        use_walls=use_walls,
        render_mode=render_mode,
        grid_size=grid_size,
        max_steps=max_steps
    )


def get_environment_defaults():
    """
    return a dictionary of default environment parameters.
    useful for interactive mode to display defaults.
    """
    return {
        "bot_difficulty": 0,
        "use_walls": False,
        "render_mode": None,
        "grid_size": 600,
        "max_steps": 3000
    }


def get_environment_options():
    """
    return a dictionary describing valid options for each parameter.
    useful for interactive mode to display choices.
    """
    return {
        "bot_difficulty": {
            "type": "int",
            "choices": [0, 1, 2, 3],
            "descriptions": {
                0: "zombie (slow, stops when close, bad aim)",
                1: "noob (normal speed, gets stuck on walls, bad aim)",
                2: "average (decent aim, standard movement, slight delay)",
                3: "pro (perfect aim, slides around corners, instant fire)"
            }
        },
        "use_walls": {
            "type": "bool",
            "choices": [True, False],
            "descriptions": {
                True: "obstacles enabled in arena",
                False: "open arena with no obstacles"
            }
        },
        "render_mode": {
            "type": "str",
            "choices": ["none", "human", "rgb_array"],
            "descriptions": {
                "none": "no rendering (fastest, for training)",
                "human": "render to screen (for watching)",
                "rgb_array": "return pixel data (for recording)"
            }
        },
        "grid_size": {
            "type": "int",
            "range": [200, 2000],
            "description": "arena size in pixels (width and height)"
        },
        "max_steps": {
            "type": "int",
            "range": [100, 100000],
            "description": "maximum steps before episode timeout"
        }
    }