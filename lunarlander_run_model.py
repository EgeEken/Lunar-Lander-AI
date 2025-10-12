import gymnasium as gym
import torch
import pygame as pg
import torch.nn as nn
import os

# ======= HYPERPARAMETERS ===========
# Model 
filename = "BEST_MODEL.pth"  # Path to save/load the model
#filename = os.path.join("temp", "BEST_MODEL_copy.pth_temp_180")
save_10ths = True  # Whether to save the model at every 10% of training progress
layer_count = 2
layer_size = 128

# Training
num_episodes = 2000     # Number of episodes to train the model
learning_rate = 1e-3    # Learning rate (how much to update the model at each training step)
gamma = 0.99            # Discount factor (how much future rewards are valued compared to immediate rewards)
epsilon = 1.0           # Exploration rate (how often to explore random actions vs exploit known information)
epsilon_decay = 0.85    # Decay rate for exploration probability per episode
epsilon_min = 0.01      # Minimum exploration probability
batch_size = 128         # Number of experiences to sample for each training step

# Replay buffer
replay_buffer = []
max_buffer_size = 1000  # Maximum size of the replay buffer

# Verbosity
verbose = False
extra_verbose = False
human_view = False
episode_count = 500

# Rewards
custom = True
# ====================================

# Define the neural network model
class LunarLanderModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LunarLanderModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, layer_size),
            nn.ReLU(),
            *[layer for _ in range(layer_count - 1) for layer in (nn.Linear(layer_size, layer_size), nn.ReLU())],
            nn.Linear(layer_size, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class CustomLunarLanderEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Call the original step method
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Modify the reward logic
        if terminated:
            # Adjust the reward for near-perfect landings
            if reward != 100:
            
                if abs(reward) < 1.0:
                    print("NOT STOPPED PENALTY")
                    reward -= 50 # heavy penalty for not stopping 


                if observation[6] and observation[7]:  # Both legs in contact
                    reward += 25  # Reward for landing with both legs
                elif observation[6] or observation[7]:  # One leg in contact
                    reward += 5   # Reward for landing with one leg
                
                if abs(observation[0]) < 0.2: # X position in center
                    reward += 20  # Reward for landing on center
                    if abs(observation[0]) < 0.1: # X position in center
                        reward += 30  # Additional reward for very close to center

                if abs(observation[2]) < 0.1: # X velocity low
                    reward += 10  # Reward for low horizontal speed

                if abs(observation[3]) < 0.2: # Y velocity low
                    reward += 20  # Reward for low vertical speed
                    if abs(observation[3]) < 0.1: # Y velocity very low
                        reward += 20  # Additional reward for very low vertical speed

                if abs(observation[4]) < 0.1: # Angle low
                    reward += 30  # Reward for small angle

                if abs(observation[5]) < 0.1: # Angular velocity low
                    reward += 20  # Reward for low angular speed

        elif truncated:
            print("TRUNCATED PENALTY")
            reward -= 50  # Big penalty for being truncated

        else:
            penalty = 0.0
            # Add penalty for large angle before termination
            if abs(observation[4]) > 0.3: # Angle threshold
                penalty += (abs(observation[4]) - 0.3) / (1.0 - 0.3) * 5  # Penalty for large angle
                if extra_verbose and human_view:
                    print(f"Large angle penalty applied: {observation[4]:.2f}, {penalty:.2f}")
                # Penalty for large angle proportional to angle

            # Add penalty for high angular velocity before termination
            if abs(observation[5]) > 0.9: # Angular velocity threshold
                penalty += (abs(observation[5]) - 0.9) / (3.0 - 0.9) * 5  # Penalty for high angular velocity
                if extra_verbose and human_view:
                    print(f"High angular velocity penalty applied: {observation[5]:.2f}, {penalty:.2f}")
                # Penalty for high angular velocity proportional to angular velocity

            # Add penalty for high vertical speed before termination
            if abs(observation[3]) > 0.5: # Vertical speed threshold
                penalty += (abs(observation[3]) - 0.5) / (1.5 - 0.5) * 5  # Penalty for high vertical speed
                if extra_verbose and human_view:
                    print(f"High vertical speed penalty applied: {observation[3]:.2f}, {penalty:.2f}")
                # Penalty for high vertical speed proportional to speed

            # Add penalty for going up before termination
            if observation[3] > 0.1: # Going up threshold
                penalty += (observation[3] - 0.1) / (1.0 - 0.1) * 5  # Penalty for going up
                if extra_verbose and human_view:
                    print(f"Going up penalty applied: {observation[3]:.2f}, {penalty:.2f}")
                # Penalty for going up proportional to vertical speed

            
            extra_reward = 0.0
            # Extra reward for being near center before termination
            if abs(observation[0]) < 0.2: # X position in center
                extra_reward += (0.2 - abs(observation[0])) / 0.2 * 5  # Extra reward for being near center
                if extra_verbose and human_view:
                    print(f"Near center extra reward applied: {observation[0]:.2f}, {extra_reward:.2f}")
                # Extra reward for being near center proportional to distance from center

            reward -= penalty
            reward += extra_reward

        return observation, int(reward), terminated, truncated, info


# Initialise the environment and model
if custom:
    if human_view:
        env = CustomLunarLanderEnv(gym.make("LunarLander-v3", render_mode="human"))
    else:
        env = CustomLunarLanderEnv(gym.make("LunarLander-v3"))
else:
    if human_view:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3")

model_path = filename # Path to your trained model

# Load your trained model
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model = LunarLanderModel(input_dim, output_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

# Reset the environment to generate the first observation
observation, info = env.reset()
pg.init()
results = []
while human_view or len(results) < episode_count:
    for event in pg.event.get():
        if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            env.close()
            pg.quit()

    # action 0: do nothing, 1: fire right engine, 2: fire main engine, 3: fire left engine
    action = model(torch.tensor(observation, dtype=torch.float32).unsqueeze(0)).argmax().item()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        print("Reward:", reward)
        observation, info = env.reset()
        results.append(reward)


print(f"{len(results)} episodes run.")
print(f"Overall average reward: {sum(results)/len(results) if results else 0:.2f}")
print(f"100 reward rate: {sum(1 for r in results if r == 100)/len(results)*100 if results else 0:.2f}%")