import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# ======= HYPERPARAMETERS ===========
# Model
filename = "BEST_MODEL_copy.pth"  # Path to save/load the model
save_10ths = True  # Whether to save the model at every 10% of training progress
layer_count = 2
layer_size = 128

# Training
num_episodes = 200     # Number of episodes to train the model
learning_rate = 2e-4    # Learning rate (how much to update the model at each training step)
gamma = 0.99            # Discount factor (how much future rewards are valued compared to immediate rewards)
epsilon = 1.0           # Exploration rate (how often to explore random actions vs exploit known information)
epsilon_decay = 0.85    # Decay rate for exploration probability per episode
epsilon_min = 0.01      # Minimum exploration probability
batch_size = 128         # Number of experiences to sample for each training step

# Replay buffer
replay_buffer = []
max_buffer_size = 1000  # Maximum size of the replay buffer

# Verbosity
verbose = True
extra_verbose = False
human_view = False

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


input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
# if theres already the searched model, load it and train further
model = LunarLanderModel(input_dim, output_dim)
if os.path.exists(filename):
    model.load_state_dict(torch.load(filename))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return model(state).argmax().item()

def train_model():
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


rewards = []
overall_average_rewards = []

# Training loop
start = time.time()
for episode in range(num_episodes):
    state, _ = env.reset()

    while True:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > max_buffer_size:
            replay_buffer.pop(0)

        train_model()

        state = next_state

        if done:
            end = time.time()
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    rewards.append(reward)
    overall_average_rewards.append(np.mean(rewards))
    if extra_verbose:
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {reward}, Time taken: {end - start:.2f} seconds")
    if (not extra_verbose) and verbose:
        if reward < -0.5:
            print("|", end="", flush=True)
        elif reward < 20:
            print("X", end="", flush=True)
        elif reward < 99:
            print("O", end="", flush=True)
        else:
            print("%", end="", flush=True)
    # 10 times out of the num_episodes, show the average reward for the last however many episodes were in the last 10% slice
    if verbose and (num_episodes > 20) and (episode != 0 and episode != num_episodes - 1) and ((episode + 1) % (num_episodes // 10) == 0):
        avg_reward = np.mean([r for r in rewards[-(num_episodes // 10):]])
        print(f"\nAverage reward (last {num_episodes // 10} episodes): {avg_reward}, last {num_episodes // 10} rewards: {rewards[-(num_episodes // 10):]}")
        print(f"Average reward (all {episode+1} episodes): {overall_average_rewards[-1]}, Time taken: {end - start:.2f} seconds\n")
    if save_10ths and (episode != 0 and episode != num_episodes - 1) and ((episode + 1) % (num_episodes // 10) == 0):
        torch.save(model.state_dict(), os.path.join("temp", f"{filename}_temp_{episode+1}"))
        if verbose:
            print(f"Model saved as /temp/{filename}_temp_{episode+1}")


# Final average reward after all episodes
print(f"Training completed over {num_episodes} episodes in {end - start:.2f} seconds")
if verbose:
    if (num_episodes > 20):
        print(f"\nAverage reward (last {num_episodes // 10} episodes): {avg_reward:.2f}, last {num_episodes // 10} rewards: {rewards[-(num_episodes // 10):]}")
    print(f"Average reward (all {episode+1} episodes): {overall_average_rewards[-1]:.2f}, Time taken: {end - start:.2f} seconds\n")

# Save the trained model
torch.save(model.state_dict(), filename)
print(f"Model saved as {filename}")

# Plot the rewards and the moving average with 10 stops at each 10% interval
plt.scatter(range(len(rewards)), rewards, s=10, label='Rewards')

# Plot the moving average with reduced opacity
moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
plt.plot(range(len(moving_avg)), moving_avg, color='orange', alpha=0.7, label='Moving Average (window=10)')

# Plot overall average at the point of each episode with reduced opacity
plt.plot(range(len(overall_average_rewards)), overall_average_rewards, color='red', alpha=0.7, label='Overall Average')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.xticks(np.arange(0, num_episodes, step=max(1, num_episodes // 10), dtype=int))
plt.title('Rewards over Episodes')
plt.legend()
plt.show()

env.close()
