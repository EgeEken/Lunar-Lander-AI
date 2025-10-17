"""The goal of this project is to run DDPG and TD3 on LunarLander and to investigate the impact of over-estimation bias 
on performance. You can use any RL library, such as Stable baselines 3, Tianshu or CleanRL, but any other choice is allowed. 
"""
"""Your report is a four pages PDF which should contain your learning curves and your empirical study of the effect of over-estimation
 bias on performance. Think of trying to establish statistically valid conclusions. You may also provide a link to a 
 public github project where I can find your sources.
"""
import numpy as np 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MLP(nn.Module):

    def __init__(self, input_size, output_size, config=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, config),
            nn.ReLU(),
            nn.Linear(config, config),
            nn.ReLU(),
            nn.Linear(config, output_size),
        )

    def forward(self, x):
        return self.net(x)



class Actor(nn.Module): 
    
    def __init__(self):
        super().__init__()
        self.actor_network = MLP(8, 2, 256).to(device)
        self.actor_target_network = MLP(8, 2, 256).to(device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=1e-4)
        self.tanh = nn.Tanh()
        self.noise = 0.2
    def estimate_best_action_env(self, state):
        a =  self.tanh(self.actor_network(torch.as_tensor(state, dtype=torch.float32, device=device))) 
        
        a += self.noise * torch.randn_like(a)
        return a.clamp(-1, 1)
  
    
    def estimate_best_action_training(self, state):
        return self.tanh(self.actor_network(torch.as_tensor(state, dtype=torch.float32, device=device))).clamp(-1,1)

    def estimate_best_action_target(self, state):
        a =  self.tanh(self.actor_target_network(torch.as_tensor(state, dtype=torch.float32, device=device)))
        return a
        
    def update(self,batch,critic):
        S, A, S2, R, D = batch             
        Pi = self.tanh(self.actor_network(S))   
        Q  = critic.estimate_q_value_1(S, Pi)   
        loss = -Q.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        self.update_networks()
            

        
    def update_networks(self,tau=0.005):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Critic(nn.Module):
    actor = None
    def __init__(self):
        super().__init__()
        self.critic_1_network = MLP(10, 1, 256).to(device)
        self.critic_2_network = MLP(10, 1, 256).to(device)
        self.critic_1_target_network = MLP(10, 1, 256).to(device)
        self.critic_2_target_network = MLP(10, 1, 256).to(device)
        self.critic_1_target_network.load_state_dict(self.critic_1_network.state_dict())
        self.critic_2_target_network.load_state_dict(self.critic_2_network.state_dict())
        self.discount_factor = 0.99
        self.alpha = 0.1
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1_network.parameters(), lr=1e-3)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2_network.parameters(), lr=1e-3)

    def _to_batch(self,x):
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return t if t.ndim > 1 else t.unsqueeze(0)

    def estimate_q_value_1(self, state, action):
        S = self._to_batch(state); A = self._to_batch(action)
        return self.critic_1_network(torch.cat([S, A], dim=-1))  

    def estimate_q_value_2(self, state, action):
        S = self._to_batch(state); A = self._to_batch(action)
        return self.critic_2_network(torch.cat([S, A], dim=-1))

    def estimate_q_value_target_1(self, state, action):
        S = self._to_batch(state); A = self._to_batch(action)
        return self.critic_1_target_network(torch.cat([S, A], dim=-1))
    def estimate_q_value_target_2(self, state, action):
        S = self._to_batch(state); A = self._to_batch(action)
        return self.critic_2_target_network(torch.cat([S, A], dim=-1))
    

    def update(self,batch,actor):
    
        initial_state, action , observation, reward, terminated = batch
        
        with torch.no_grad():
            best_action = actor.estimate_best_action_target(observation)
            noise = (torch.randn_like(best_action) * actor.noise).clamp(-0.5, 0.5) 
            best_action = (best_action + noise).clamp(-1.0, 1.0)
            S = self._to_batch(observation); A = self._to_batch(best_action)
            q1 = self.critic_1_target_network(torch.cat([S, A], dim=-1))
            q2 = self.critic_2_target_network(torch.cat([S, A], dim=-1))
            evaluate_future_action = torch.min(q1,q2) * (1 - terminated)
            target_q = reward + self.discount_factor * evaluate_future_action
        predicted_q1 = self.estimate_q_value_1(initial_state,action)
        predicted_q2 = self.estimate_q_value_2(initial_state,action)
        loss1 = F.mse_loss(predicted_q1,target_q)
        loss2 = F.mse_loss(predicted_q2,target_q)

        self.critic_1_optimizer.zero_grad()
        loss1.backward() #we need to update the true critic network not the target
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss2.backward() #we need to update the true critic network not the target
        self.critic_2_optimizer.step()

        

    def update_networks(self,tau=0.005):
        for target_param, param in zip(self.critic_1_target_network.parameters(), self.critic_1_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target_network.parameters(), self.critic_2_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class TD_train():
    def __init__(self,batch_size):
        self.replay_buffer = []
        self.actor = Actor()
        self.critic = Critic()
        

        self.capacity = 100000
        self.batch_size = batch_size    

    def load_batch(self, batch_size):
        idx = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        S, A, S2, R, D = zip(*[self.replay_buffer[i] for i in idx])
        S  = torch.as_tensor(np.stack(S),  dtype=torch.float32, device=device)
        A  = torch.as_tensor(np.stack(A),  dtype=torch.float32, device=device)
        S2 = torch.as_tensor(np.stack(S2), dtype=torch.float32, device=device)
        R  = torch.as_tensor(np.array(R)[:,None], dtype=torch.float32, device=device)
        D  = torch.as_tensor(np.array(D)[:,None], dtype=torch.float32, device=device)
        return S, A, S2, R, D

    
    def replay_buffer_add(self,initial_state, action , obs, reward, terminated):
        self.replay_buffer.append((np.asarray(initial_state), np.asarray(action), np.asarray(obs), np.asarray(reward), np.asarray(terminated)))
        if len(self.replay_buffer) > self.capacity:
            self.replay_buffer.pop(0)


    def save(self, path, meta=None):
        ckpt = {
            # ---- actor ----
            'actor': self.actor.actor_network.state_dict(),
            'actor_target': self.actor.actor_target_network.state_dict(),
            'actor_opt': self.actor.actor_optimizer.state_dict(),

            # ---- twin critics ----
            'critic1': self.critic.critic_1_network.state_dict(),
            'critic2': self.critic.critic_2_network.state_dict(),
            'critic1_target': self.critic.critic_1_target_network.state_dict(),
            'critic2_target': self.critic.critic_2_target_network.state_dict(),
            'critic1_opt': self.critic.critic_1_optimizer.state_dict(),
            'critic2_opt': self.critic.critic_2_optimizer.state_dict(),

            # ---- extras ----
            'replay_buffer': self.replay_buffer,
            'meta': meta or {},
        }
        torch.save(ckpt, path)


    def load(self, path, map_location=device):
        ckpt = torch.load(path, map_location=map_location)

        # ---- actor ----
        self.actor.actor_network.load_state_dict(ckpt['actor'])
        self.actor.actor_target_network.load_state_dict(ckpt['actor_target'])
        self.actor.actor_optimizer.load_state_dict(ckpt['actor_opt'])

        # ---- twin critics ----
        self.critic.critic_1_network.load_state_dict(ckpt['critic1'])
        self.critic.critic_2_network.load_state_dict(ckpt['critic2'])
        self.critic.critic_1_target_network.load_state_dict(ckpt['critic1_target'])
        self.critic.critic_2_target_network.load_state_dict(ckpt['critic2_target'])
        self.critic.critic_1_optimizer.load_state_dict(ckpt['critic1_opt'])
        self.critic.critic_2_optimizer.load_state_dict(ckpt['critic2_opt'])

        # ---- extras ----
        self.replay_buffer = ckpt.get('replay_buffer', [])
        return ckpt.get('meta', {})


import time
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

env = gym.make("LunarLanderContinuous-v2")
obs, info = env.reset(seed=0)
initial_state = obs
batch_size = 64
ddpg = TD_train(batch_size)
ddpg_warmup = 5000
total_steps = 0

episode_rewards = []       
avg_rewards_smoothed = []  
window = 10                
policy_delay = 2 
try:
    while total_steps < 50000:  
        terminated = truncated = False
        obs, info = env.reset()
        initial_state = obs
        episode_reward = 0.0

        while not (terminated or truncated):
            # Choose action
            if total_steps < ddpg_warmup:
                action = env.action_space.sample()
            else:
                action = ddpg.actor.estimate_best_action_env(initial_state).detach().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Store experience + train
            done = terminated or truncated
            ddpg.replay_buffer_add(initial_state, action, obs, reward, done )
            if total_steps >= ddpg_warmup and len(ddpg.replay_buffer) >= ddpg.batch_size:
                batch = ddpg.load_batch(ddpg.batch_size)
                ddpg.critic.update(batch,ddpg.actor)
                if total_steps % policy_delay == 0:
                    ddpg.actor.update(batch,ddpg.critic)
                    ddpg.critic.update_networks()

            total_steps += 1
            initial_state = obs
            if total_steps % 1000 == 0:
                print(f"Total steps: {total_steps}")

        # --- end of episode ---
        episode_rewards.append(episode_reward)
        # moving average for smoother curve
        avg_reward = np.mean(episode_rewards[-window:])
        avg_rewards_smoothed.append(avg_reward)


except KeyboardInterrupt:
    pass
finally:
    env.close()

ddpg.save("td_llc.pt", meta={'total_steps': total_steps,
                               'episode_rewards': episode_rewards})




import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
def moving_average(x, window):
    if len(x) == 0: 
        return []
    window = max(1, int(window))
    cumsum = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad front so it has same length for nicer plotting
    pad = [np.nan] * (window - 1)
    return pad + ma.tolist()

def plot_learning_curves(episode_rewards, smooth_window=10, title="Training Performance", out_png="learning_curve.png"):
    ep = np.arange(1, len(episode_rewards) + 1)
    ma = moving_average(episode_rewards, smooth_window)

    plt.figure(figsize=(8, 5))
    plt.plot(ep, episode_rewards, label="Episode reward")
    plt.plot(ep, ma, label=f"Moving average ({smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ensure folder exists, then save
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved plot to: {out_png}")

plot_learning_curves(
    episode_rewards,
    smooth_window=10,
    title="TD3 Training on LunarLanderContinuous-v2",
    out_png="results/td3_learning_curve.png"
)
env = gym.make("LunarLanderContinuous-v2", render_mode="human")

num_episodes = 10
avg_reward = 0 
for ep in range(num_episodes):
    obs, info = env.reset()
    initial_state = obs
    done = False
    episode_reward = 0.0

    while not done:
        with torch.no_grad():
            action = ddpg.actor.estimate_best_action_training(obs).cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    print(f"Episode {ep+1}/{num_episodes} | Total reward: {episode_reward:.1f}")
    avg_reward += episode_reward
avg_reward /= num_episodes
print(f"Average Reward over {num_episodes} episodes: {avg_reward:.1f}")
env.close()
