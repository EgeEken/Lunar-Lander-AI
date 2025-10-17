# eval_td_lunarlander.py
import os
import numpy as np
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
        self.noise = 0.2  # kept for compatibility

    def estimate_best_action_env(self, state):
        a = self.tanh(self.actor_network(torch.as_tensor(state, dtype=torch.float32, device=device)))
        a += self.noise * torch.randn_like(a)
        return a.clamp(-1, 1)

    def estimate_best_action_training(self, state):
        return self.tanh(self.actor_network(torch.as_tensor(state, dtype=torch.float32, device=device))).clamp(-1, 1)

    def estimate_best_action_target(self, state):
        a = self.tanh(self.actor_target_network(torch.as_tensor(state, dtype=torch.float32, device=device)))
        return a

    def update(self, batch, critic):
        S, A, S2, R, D = batch
        Pi = self.tanh(self.actor_network(S))
        Q = critic.estimate_q_value_1(S, Pi)
        loss = -Q.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        self.update_networks()

    def update_networks(self, tau=0.005):
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

    def _to_batch(self, x):
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

    # (training-related methods kept for checkpoint compatibility)
    def update(self, batch, actor):
        raise RuntimeError("This script is evaluation-only; no training step is allowed.")

    def update_networks(self, tau=0.005):
        for target_param, param in zip(self.critic_1_target_network.parameters(), self.critic_1_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target_network.parameters(), self.critic_2_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class TD_train():
    def __init__(self, batch_size):
        self.replay_buffer = []
        self.actor = Actor()
        self.critic = Critic()
        self.capacity = 100000
        self.batch_size = batch_size

    def load(self, path, map_location=device):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.actor_network.load_state_dict(ckpt['actor'])
        self.actor.actor_target_network.load_state_dict(ckpt['actor_target'])
        self.actor.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic.critic_1_network.load_state_dict(ckpt['critic1'])
        self.critic.critic_2_network.load_state_dict(ckpt['critic2'])
        self.critic.critic_1_target_network.load_state_dict(ckpt['critic1_target'])
        self.critic.critic_2_target_network.load_state_dict(ckpt['critic2_target'])
        self.critic.critic_1_optimizer.load_state_dict(ckpt['critic1_opt'])
        self.critic.critic_2_optimizer.load_state_dict(ckpt['critic2_opt'])
        self.replay_buffer = ckpt.get('replay_buffer', [])
        return ckpt.get('meta', {})

# ----------------------------- Evaluation only -----------------------------
def evaluate(ckpt_path="td_llc.pt", num_episodes=10, render=True, seed=0):
    import gymnasium as gym

    render_mode = "human" if render else None
    env = gym.make("LunarLanderContinuous-v2", render_mode=render_mode)

    # build policy and load weights
    ddpg = TD_train(batch_size=64)
    meta = ddpg.load(ckpt_path)
    ddpg.actor.actor_network.eval()  # no dropout/bn anyway, but explicit

    avg_reward = 0.0
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
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

    avg_reward /= max(1, num_episodes)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.1f}")
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TD weights on LunarLanderContinuous-v2 (no training).")
    parser.add_argument("--ckpt", type=str, default="td_llc.pt", help="Path to checkpoint saved by your training script.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--no-render", action="store_true", help="Disable human rendering.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for env resets.")
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        num_episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed,
    )
