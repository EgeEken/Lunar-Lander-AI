#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size),
        )
    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.actor_network = MLP(8, 2, 64).to(device)
        self.tanh = nn.Tanh()
    @torch.no_grad()
    def act(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.tanh(self.actor_network(s))
        return a.clamp(-1, 1).cpu().numpy()

def deadzone_eval(action: np.ndarray,
                  main_pos_eps: float = 0.15,  # ↑ zero tiny upward thrust
                  main_neg_eps: float = 0.02,  # keep tiny negative (down) as is
                  side_eps: float = 0.05) -> np.ndarray:
    a = action.astype(np.float32).copy()
    # main engine (index 0)
    if 0.0 < a[0] < main_pos_eps:
        a[0] = 0.0
    if -main_neg_eps < a[0] < 0.0:
        a[0] = 0.0
    # side engine (index 1)
    if abs(a[1]) < side_eps:
        a[1] = 0.0
    return a

def load_actor(ckpt_path: str, device: torch.device) -> Actor:
    actor = Actor(device)
    # Try: (1) full training checkpoint dict; (2) actor-only state_dict
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "actor" in ckpt:
            state = ckpt["actor"]
        else:
            state = ckpt  # assume it's already a state_dict
    except TypeError:
        # Older torch without weights_only kw
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt
    actor.actor_network.load_state_dict(state)
    actor.eval()
    return actor

def run_eval(env_id: str, ckpt_path: str, episodes: int, render_mode: str,
             seed: int, deadzone_eps: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_actor(ckpt_path, device)
    env = gym.make(env_id, render_mode=render_mode)

    returns = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done, total_r = False, 0.0
        while not done:
            action = actor.act(obs)
            
            
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += r
        print(f"Episode {ep+1}/{episodes} — Return: {total_r:.1f}")
        returns.append(total_r)
    env.close()
    print(f"Average return: {np.mean(returns):.1f}")

def main():
    p = argparse.ArgumentParser(description="Run saved policy on LunarLanderContinuous-v3.")
    p.add_argument("--ckpt", type=str, default="ddpg_llc.pt")
    p.add_argument("--env", type=str, default="LunarLanderContinuous-v3")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadzone", type=float, default=0.05)
    p.add_argument("--render", type=str, default="human", choices=["human","rgb_array"])
    args = p.parse_args()
    run_eval(args.env, args.ckpt, args.episodes, args.render, args.seed, args.deadzone)

if __name__ == "__main__":
    main()
