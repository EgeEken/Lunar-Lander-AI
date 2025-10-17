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

# ===== PARAMETERS =====
generation_count = 300   # Number of generations (iterations)

# Essentially the size of the model
population_size = 40    # Population size

# Essentially the learning rate
selection_rate = 0.3    # Proportion of population to keep
mutation_rate = 0.2     # Mutation rate
mutation_strength = 0.5 # Standard deviation of mutation noise 

verbose = True          # Print info about each generation
extra_verbose = False   # Print detailed info about each individual
# ===============================

# ===== SIMPLE LINEAR GENETIC ALGORITHM =====
class SimpleGAAgent:
    def __init__(self):
        self.genome = np.random.uniform(-1, 1, (2, 8))

    def get_action(self, obs):
        # Simple linear policy: action = W * obs
        action = np.dot(self.genome, obs)
        return np.clip(action, -1, 1)

    def mutate(self, mutation_rate, mutation_strength):
        mutation_mask = np.random.rand(*self.genome.shape) < mutation_rate
        mutations = np.random.normal(0, mutation_strength, self.genome.shape)
        self.genome += mutation_mask * mutations
        self.genome = np.clip(self.genome, -1, 1)

    def save_state_dict(self, filename):
        # Ensure the directory exists
        os.makedirs("params", exist_ok=True)
        try:
            np.save(Path("params", filename), self.genome)
        except Exception as e:
            print(f"Error saving state dict: {e}")

    def load_state_dict(self, filename):
        try:
            self.genome = np.load(Path("params", filename), allow_pickle=True)
        except Exception as e:
            print(f"Error loading state dict: {e}")
# ==========================================

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

def run_agent_model(agent_model, human_view=True, max_episodes=10, verbose=True, extra_verbose=False):
    if human_view:
        env = gym.make('LunarLanderContinuous-v3', render_mode='human')
        pg.font.init()
        font = pg.font.SysFont('Futura', 20)
    else:
        env = gym.make('LunarLanderContinuous-v3')
    obs, info = env.reset()
    done = False
    ep_count = 0
    rewards = []
    successes = []
    total_reward = 0.0
    while (not done) and (ep_count < max_episodes):
        if human_view:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        done = True

        action = agent_model.get_action(obs)
        average_reward = np.mean(rewards) if rewards else 0.0
        if human_view:
            lines = [
            f"Episode {ep_count} / {max_episodes}",
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
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            ep_count += 1
            if reward == 100:
                successes.append(1)
            else:
                successes.append(0)
            rewards.append(total_reward)
            if verbose:
                if extra_verbose or (ep_count % max(1, max_episodes // 10) == 0):
                    print(f"Episode {ep_count} - Reward: {total_reward:.2f} - Average Reward: {np.mean(rewards):.2f} - Success Rate: {100 * np.mean(successes):.2f}%\n")
            env.reset()
            total_reward = 0.0
    env.close()
    pg.quit()
    print(f"Finished running {ep_count} episodes.")
    print(f"Average Reward over {ep_count} episodes: {np.mean(rewards):.2f}")
    print(f"Success Rate over {ep_count} episodes: {100 * np.mean(successes):.2f}%")
    return rewards, successes

loaded_agent = SimpleGAAgent()
loaded_agent.load_state_dict("BEST_SIMPLE_GA_AGENT.npy")

# run the model from the best generation
max_episodes = 20
best_agent_rewards = run_agent_model(loaded_agent, human_view=True, max_episodes=max_episodes, verbose=True, extra_verbose=True)
