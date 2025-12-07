"""Utility to visualize RL quantization training logs.

Parses terminal logs that contain lines like:
  Episode 1/50 | Reward: 1.1175 | PPL: 16.59 | Memory: 62.33%
and produces a PNG with Reward, PPL, and Memory curves over episodes.
"""

import argparse
import os
import re
from typing import Dict, List

# Avoid OpenMP shared-memory issues in constrained environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import matplotlib

# Use non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPISODE_PATTERN = re.compile(
    r"Episode\s+(\d+)\s*/\s*(\d+)\s*\|\s*Reward:\s*([\-0-9.]+)\s*\|\s*PPL:\s*([\-0-9.]+)\s*\|\s*Memory:\s*([\-0-9.]+)%",
    re.IGNORECASE,
)


def parse_log(path: str) -> Dict[str, List[float]]:
    """Extract episode, reward, ppl, memory columns from a log file."""
    episodes: List[int] = []
    rewards: List[float] = []
    ppls: List[float] = []
    memories: List[float] = []
    total_episodes: int = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = EPISODE_PATTERN.search(line)
            if not match:
                continue

            ep, total, reward, ppl, memory = match.groups()
            episodes.append(int(ep))
            total_episodes = int(total)
            rewards.append(float(reward))
            ppls.append(float(ppl))
            memories.append(float(memory))

    if not episodes:
        raise ValueError("No episode lines were found in the log.")

    return {
        "episode": episodes,
        "total_episodes": total_episodes,
        "reward": rewards,
        "ppl": ppls,
        "memory": memories,
    }


def plot_metrics(data: Dict[str, List[float]], out_path: str) -> str:
    """Plot reward, ppl, and memory curves."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    episodes = data["episode"]

    axes[0].plot(episodes, data["reward"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward by Episode")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(episodes, data["ppl"], marker="o", color="#d62728")
    axes[1].set_ylabel("PPL")
    axes[1].set_title("Perplexity by Episode")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(episodes, data["memory"], marker="o", color="#2ca02c")
    axes[2].set_ylabel("Memory Saving (%)")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Memory Saving by Episode")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"RL Quantization Training ({len(episodes)} / {data['total_episodes']} episodes)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot RL training log metrics.")
    parser.add_argument(
        "log_file",
        help="Path to log file that contains 'Episode X/Y | Reward: ... | PPL: ... | Memory: ...' lines.",
    )
    parser.add_argument(
        "--out",
        default="analysis_results/training_metrics.png",
        help="Where to save the generated figure (PNG).",
    )
    args = parser.parse_args()

    data = parse_log(args.log_file)
    out_path = plot_metrics(data, args.out)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
