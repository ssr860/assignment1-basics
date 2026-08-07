# >>>>..........

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = args.run_dir / "metrics.jsonl"
    output_path = args.run_dir / "loss_curve.png"

    records = []

    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    colors = {
        "train": "tab:blue",
        "validation": "tab:orange",
    }

    for split in ("train", "validation"):
        split_records = [
            record
            for record in records
            if record["split"] == split
        ]

        if not split_records:
            continue

        steps = [
            record["step"]
            for record in split_records
        ]
        wall_times = [
            record["wall_time_seconds"] / 60
            for record in split_records
        ]
        losses = [
            record["loss"]
            for record in split_records
        ]

        axes[0].plot(
            steps,
            losses,
            label=split,
            color=colors[split],
        )

        axes[1].plot(
            wall_times,
            losses,
            label=split,
            color=colors[split],
        )

    axes[0].set_xlabel("Gradient steps")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss vs. gradient steps")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Wall-clock time (minutes)")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Loss vs. wall-clock time")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    print(f"Saved loss curve to {output_path}")


if __name__ == "__main__":
    main()