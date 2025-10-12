import gymnasium as gym
import pygame

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
                if observation[6] and observation[7]:  # Both legs in contact
                    reward += 25  # Reward for landing with both legs
                elif observation[6] or observation[7]:  # One leg in contact
                    reward += 5   # Reward for landing with one leg
                
                if abs(observation[0]) < 0.2: # X position in center
                    reward += 30  # Reward for landing on center

                if abs(observation[2]) < 0.1: # X velocity low
                    reward += 10  # Reward for low horizontal speed

                if abs(observation[3]) < 0.1: # Y velocity low
                    reward += 40  # Reward for low vertical speed

                if abs(observation[4]) < 0.1: # Angle low
                    reward += 20  # Reward for small angle

                if abs(observation[5]) < 0.1: # Angular velocity low
                    reward += 10  # Reward for low angular speed

        return observation, reward, terminated, truncated, info


# Initialise the environment and model
#env = gym.make("LunarLander-v3")
#env = gym.make("LunarLander-v3", render_mode="human")

# Use the custom wrapper
env = CustomLunarLanderEnv(gym.make("LunarLander-v3", render_mode="human"))

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(2000):
    # Use keyboard inputs to control the agent
    pygame.init()
    keys = pygame.key.get_pressed()

    # Map keyboard inputs to actions
    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 3
    elif keys[pygame.K_UP]:
        action = 2
    else:
        action = 0

    # action 0: do nothing, 1: fire right engine, 2: fire main engine, 3: fire left engine

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        #print(observation)
        print(f"observation[0] (x position): {observation[0]}")
        print(f"observation[1] (y position): {observation[1]}")
        print(f"observation[2] (x velocity): {observation[2]}")
        print(f"observation[3] (y velocity): {observation[3]}")
        print(f"observation[4] (angle): {observation[4]}")
        print(f"observation[5] (angular velocity): {observation[5]}")
        print(f"observation[6] (left leg contact): {observation[6]}")
        print(f"observation[7] (right leg contact): {observation[7]}")
        #print(info)
        print(f"reward: {reward}")
        observation, info = env.reset()

env.close()