import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ppo import Config, compute_gae, train  # noqa: E402


def test_gae_hand_worked():
    # One env, 3 steps. gamma=0.5, lambda=0.5 so gamma*lambda = 0.25.
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.tensor([[2.0], [4.0], [6.0]])
    dones = torch.zeros(3, 1)
    next_value, next_done = torch.tensor([8.0]), torch.tensor([0.0])
    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, 0.5, 0.5)
    # delta2 = 1 + 0.5*8 - 6 = -1            A2 = -1
    # delta1 = 1 + 0.5*6 - 4 =  0            A1 = 0 + 0.25*(-1) = -0.25
    # delta0 = 1 + 0.5*4 - 2 =  1            A0 = 1 + 0.25*(-0.25) = 0.9375
    assert torch.allclose(adv.squeeze(), torch.tensor([0.9375, -0.25, -1.0]))
    assert torch.allclose(ret.squeeze(), torch.tensor([2.9375, 3.75, 5.0]))


def test_gae_cuts_at_episode_boundary():
    # dones[1]=1 means obs[1] starts a new episode, so step 0 must not bootstrap from it.
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[3.0], [5.0]])
    dones = torch.tensor([[0.0], [1.0]])
    adv, _ = compute_gae(rewards, values, dones, torch.tensor([7.0]), torch.tensor([0.0]), 0.9, 0.9)
    assert abs(adv[0].item() - (1.0 - 3.0)) < 1e-6   # delta0 = r - V, no bootstrap


def test_smoke_tiny_run(tmp_path):
    out = tmp_path / "log.json"
    res = train(Config(total_timesteps=1024, num_envs=2, num_steps=64, out=str(out)), log=lambda *_: None)
    assert out.exists()
    assert len(res["returns"]) > 0
    assert all(r >= 1 for r in res["returns"])
