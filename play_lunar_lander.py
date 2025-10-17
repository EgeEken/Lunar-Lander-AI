# ===============================
import os
from pathlib import Path
import time
import numpy as np 
import matplotlib.pyplot as plt

# ===============================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
print(f"Using torch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===============================
import gymnasium as gym
import pygame as pg
print(f"Using gym version: {gym.__version__}")
print(f"Using pygame version: {pg.__version__}")
# ===============================


def get_color(score):
    """
    Optional function to color the text based on the score.
    green >= 90
    red-yellow-green gradient (-90, 90)
    red <= -90
    """
    if score >= 90:
        return (0, 255, 0)  # Green
    elif score <= -90:
        return (255, 0, 0)  # Red
    else:
        # Gradient from red to green
        ratio = (score + 90) / 180
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        return (r, g, 0)


env = gym.make('LunarLanderContinuous-v3', render_mode='human')
pg.font.init()
font = pg.font.SysFont('Futura', 20)
obs, info = env.reset()
done = False
ep_count = 0
rewards = []
total_reward = 0.0

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                done = True

    keys = pg.key.get_pressed()
    action = np.array([0.0, 0.0], dtype=np.float32)
    average_reward = np.mean(rewards) if rewards else 0.0
    lines = [
    f"Episode: {ep_count}",
    f"Current Reward: {total_reward:.2f}",
    f"Last Reward: {rewards[-1] if rewards else 0:.2f}",
    f"Average Reward: {average_reward:.2f}"
    ]
    screen = pg.display.get_surface()  # Get the current display surface
    if screen:
        y_offset = 10  # Start drawing from y=10
        for line in lines:
            color = get_color(total_reward) if "Current Reward" in line else (255, 255, 255)
            text_surface = font.render(line, True, color)
            screen.blit(text_surface, (10, y_offset))  # Top-left corner of the screen
            y_offset += text_surface.get_height() + 5  # Add spacing between lines

    pg.display.update()  # Update the display


    if keys[pg.K_UP]:
        action[0] = 1.0  # Main engine
    if keys[pg.K_LEFT]:
        action[1] = -1.0  # Left side engine
    if keys[pg.K_RIGHT]:
        action[1] = 1.0  # Right side engine

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        ep_count += 1
        rewards.append(total_reward)
        print(f"\nEpisode {ep_count} - Reward: {total_reward:.2f}\n")
        env.reset()
        total_reward = 0.0

env.close()
pg.quit()