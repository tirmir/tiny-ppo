"""PPO from scratch for discrete-action Gymnasium environments (default CartPole-v1).

One file, no framework. Details included: GAE, clipped surrogate loss, clipped
value loss, entropy bonus, advantage normalisation, orthogonal init, linear
learning-rate annealing, gradient clipping, vectorised envs.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 1
    total_timesteps: int = 100_000
    num_envs: int = 4
    num_steps: int = 128          # rollout length per env
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    out: str = ""                  # optional path for a JSON log of episode returns


def layer_init(layer: nn.Linear, std: float = 2 ** 0.5, bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Separate actor and critic MLPs (two hidden layers of 64, tanh)."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # small std on the last actor layer keeps the first policy close to uniform
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        dist = Categorical(logits=self.actor(obs))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)


def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """Generalised Advantage Estimation.

    rewards, values, dones: tensors of shape (T, N). dones[t] is 1 if the episode
    ended on the step that produced obs[t] (i.e. obs[t] is the first obs of a new
    episode), the usual CleanRL convention. next_value / next_done: shape (N,),
    for the observation after the last step. Returns (advantages, returns).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done
            v_next = next_value
        else:
            nonterminal = 1.0 - dones[t + 1]
            v_next = values[t + 1]
        delta = rewards[t] + gamma * v_next * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def make_envs(env_id: str, num_envs: int, seed: int):
    envs = gym.make_vec(
        env_id, num_envs=num_envs, vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    return envs


def train(cfg: Config, log=print) -> dict:
    """Train and return {'steps': [...], 'returns': [...], 'wall_time': float}."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(1)  # tiny nets: threading only adds overhead

    envs = make_envs(cfg.env_id, cfg.num_envs, cfg.seed)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)
    agent = Agent(obs_dim, n_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    T, N = cfg.num_steps, cfg.num_envs
    batch_size = T * N
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = max(1, cfg.total_timesteps // batch_size)

    obs_buf = torch.zeros((T, N, obs_dim))
    act_buf = torch.zeros((T, N), dtype=torch.long)
    logp_buf = torch.zeros((T, N))
    rew_buf = torch.zeros((T, N))
    done_buf = torch.zeros((T, N))
    val_buf = torch.zeros((T, N))

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
    next_done = torch.zeros(N)
    ep_ret = np.zeros(N)
    steps_log: list[int] = []
    returns_log: list[float] = []
    global_step = 0
    start = time.time()

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # --- collect a rollout ---
        for t in range(T):
            global_step += N
            obs_buf[t], done_buf[t] = next_obs, next_done
            with torch.no_grad():
                action, logp, _, value = agent.act(next_obs)
            act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

            obs_np, reward, term, trunc, _ = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32)
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32)
            next_done = torch.as_tensor(done, dtype=torch.float32)

            ep_ret += reward
            for i in np.flatnonzero(done):
                steps_log.append(global_step)
                returns_log.append(float(ep_ret[i]))
                ep_ret[i] = 0.0

        # --- advantages ---
        with torch.no_grad():
            next_value = agent.value(next_obs)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, next_done, cfg.gamma, cfg.gae_lambda)

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret, b_val = advantages.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

        # --- optimise ---
        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch_size)
            for s in range(0, batch_size, minibatch_size):
                idx = perm[s:s + minibatch_size]
                _, new_logp, entropy, new_val = agent.act(b_obs[idx], b_act[idx])
                ratio = (new_logp - b_logp[idx]).exp()

                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # per-minibatch normalisation
                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()

                v_unclipped = (new_val - b_ret[idx]) ** 2
                v_clipped = b_val[idx] + (new_val - b_val[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b_ret[idx]) ** 2).mean()

                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if update % 20 == 0 or update == num_updates:
            recent = returns_log[-20:]
            avg = float(np.mean(recent)) if recent else float("nan")
            log(f"seed={cfg.seed} step={global_step} avg_return(last20)={avg:.1f} "
                f"elapsed={time.time() - start:.0f}s")

    envs.close()
    result = {"config": asdict(cfg), "steps": steps_log, "returns": returns_log,
              "wall_time": time.time() - start}
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(result))
    return result


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="PPO from scratch (discrete actions)")
    d = Config()
    for name, value in asdict(d).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        else:
            p.add_argument(flag, type=type(value), default=value)
    return Config(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
