# INITIAL
# 1. Create Environment
```bash
conda create -n tank_ai python=3.10 -y
```
```bash
conda activate tank_ai
```
```bash
pip install gymnasium pygame stable-baselines3 numpy tensorboard
pip install moviepy
pip install imageio imageio[ffmpeg]
```




# DAILY-USE
```bash
conda activate tank_ai
# pc 1
cd /mnt/c/Users/C/OneDrive/1-PlayGround/-01-AI-CLASS/Tank_Game

# pc 2
cd "/mnt/c/Users/chris/OneDrive/1-PlayGround/-01-AI-CLASS/Tank_Game2"
```

# Step 1: Test the game works
python play_human.py
# Play a few rounds, make sure controls feel good

# Step 2: If game is fun, train the AI
python train_agent.py
# Go get coffee, come back in 20 mins

# Step 3: Fight the trained AI
# Edit play_human.py line 6 to:
env = TankCombatEnv(render_mode="human", opponent_type="policy", 
                    opponent_policy=PPO.load("tank_basic_model"))