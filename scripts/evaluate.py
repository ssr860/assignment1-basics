from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.blocks import Transformer
from cs336_basics.evaluation import evaluate_loss
from cs336_basics.training_utilities import load_checkpoint


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CS336 language-model checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the checkpoint with the largest step number.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Defaults to val_data_path stored in config.json.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:0.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a JSON object in {config_path}")
    return config


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return -1


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint_dir = run_dir / "checkpoints"
    candidates = list(checkpoint_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {checkpoint_dir}")
    return max(candidates, key=checkpoint_step)


def build_model(
    config: dict[str, Any],
    device: torch.device,
) -> Transformer:
    dtype_name = str(config.get("dtype", "float32"))
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in config.json: {dtype_name}")

    return Transformer(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=DTYPE_MAP[dtype_name],
    )


def load_token_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Token array not found: {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D token array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Token array must contain integers, got {array.dtype}")
    return array


def main() -> None:
    args = parse_args()
    config = load_config(args.run_dir)
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)

    configured_val_path = config.get("val_data_path")
    if args.data_path is None and configured_val_path is None:
        raise ValueError(
            "No --data-path was supplied and config.json has no val_data_path"
        )
    data_path = args.data_path or Path(str(configured_val_path))

    batch_size = args.batch_size or int(config["batch_size"])
    eval_iters = args.eval_iters or int(config["eval_iters"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)) + 1
    if batch_size <= 0 or eval_iters <= 0:
        raise ValueError("batch_size and eval_iters must be positive")

    dataset = load_token_array(data_path)
    model = build_model(config, device)

    iteration = load_checkpoint(
        src=checkpoint_path,
        model=model,
        optimizer=None,
    )

    loss = evaluate_loss(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        context_length=int(config["context_length"]),
        eval_iters=eval_iters,
        device=device,
        seed=seed,
    )
    perplexity = math.exp(loss)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step:       {iteration}")
    print(f"Data:       {data_path}")
    print(f"Loss:       {loss:.6f}")
    print(f"Perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    main()