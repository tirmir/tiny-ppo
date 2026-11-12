"""Plot learning curves from the JSON logs written by `ppo.py --out`."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth_on_grid(steps, returns, grid, window=20):
    """Moving average of the last `window` episodes, read off at each grid step."""
    steps, returns = np.asarray(steps), np.asarray(returns)
    out = []
    for g in grid:
        n = np.searchsorted(steps, g, side="right")
        out.append(returns[max(0, n - window):n].mean() if n else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default="assets/learning_curve.png")
    args = ap.parse_args()

    runs = [json.loads(Path(f).read_text()) for f in args.logs]
    max_step = min(r["steps"][-1] for r in runs)
    grid = np.linspace(2000, max_step, 200)
    curves = np.stack([smooth_on_grid(r["steps"], r["returns"], grid) for r in runs])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r, c in zip(runs, curves):
        ax.plot(grid, c, alpha=0.35, lw=1, label=f"seed {r['config']['seed']}")
    ax.plot(grid, curves.mean(0), color="black", lw=2, label="mean")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return (mean of last 20 episodes)")
    ax.set_title("tiny-ppo on CartPole-v1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
