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
```

# DAILY-USE
```bash
conda activate tank_ai
cd /YOUR/SPECIFIC/FILE/PATH/Tank_Game
```


# Step 1: Test the game works
python play_human.py
# Play a few rounds, make sure controls feel good, and BOT is correct.
