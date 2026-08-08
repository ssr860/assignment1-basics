from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cs336_basics.blocks import Transformer
from cs336_basics.evaluation import evaluate_loss
from cs336_basics.experiment_logging import ExperimentLogger
from cs336_basics.layers import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.training_utilities import (
    get_batch,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)


DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CS336 Transformer language model."
    )

    # Data and experiment paths.
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--val-data-path", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("artifacts/experiments"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Defaults to a timestamp for a new run.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint path such as run_dir/checkpoints/step_0001000.pt.",
    )

    # Model configuration.
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # Optimization configuration.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-end-step", type=int, default=10_000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging, evaluation, and checkpointing.
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)

    # Runtime configuration.
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_MAP),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "max_steps",
        "decay_end_step",
        "log_interval",
        "eval_interval",
        "eval_iters",
        "checkpoint_interval",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.decay_end_step <= args.warmup_steps:
        raise ValueError("--decay-end-step must be greater than --warmup-steps")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if not 0.0 <= args.min_lr <= args.max_lr:
        raise ValueError("learning rates must satisfy 0 <= min_lr <= max_lr")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise ValueError("--beta1 and --beta2 must be in [0, 1)")
    if args.adam_eps <= 0.0:
        raise ValueError("--adam-eps must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_run_directory(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_from is not None:
        checkpoint_directory = args.resume_from.resolve().parent
        if checkpoint_directory.name != "checkpoints":
            raise ValueError(
                "--resume-from must point inside a run's checkpoints/ directory"
            )
        return checkpoint_directory.parent, True

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.experiment_root / run_name, False


def load_token_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Token array not found: {path}")

    array = np.load(path, mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D token array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"Token array must contain integers, got {array.dtype}")
    return array


def set_learning_rate(optim: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optim.param_groups:
        parameter_group["lr"] = learning_rate


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_dtype = DTYPE_MAP[args.dtype]
    run_dir, is_resuming = resolve_run_directory(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_data = load_token_array(args.train_data_path)
    val_data = load_token_array(args.val_data_path)

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=model_dtype,
    )

    optim = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optim,
            map_location=device,
        )

    if start_step >= args.max_steps:
        raise ValueError(
            f"Checkpoint is already at step {start_step}, but --max-steps is "
            f"{args.max_steps}. Choose a larger --max-steps."
        )

    logger = ExperimentLogger(
        run_dir=run_dir,
        config=vars(args),
        resume=is_resuming,
    )

    print(f"Run directory:     {run_dir}")
    print(f"Training tokens:   {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Dataset dtype:     {train_data.dtype}")
    print(f"Model dtype:       {model_dtype}")
    print(f"Device:            {device}")
    print(f"Starting step:     {start_step}")

    model.train()
    running_loss = 0.0
    steps_since_log = 0
    last_log_time = time.perf_counter()
    last_learning_rate = args.max_lr

    for step in range(start_step, args.max_steps):
        learning_rate = learning_rate_schedule(
            step=step,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            decay_end_step=args.decay_end_step,
        )
        last_learning_rate = learning_rate
        set_learning_rate(optim, learning_rate)

        inputs, targets = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        optim.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        gradient_clipping(
            parameters=model.parameters(),
            max_l2_norm=args.max_grad_norm,
        )
        optim.step()

        completed_step = step + 1
        running_loss += loss.detach().item()
        steps_since_log += 1

        if completed_step % args.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            current_time = time.perf_counter()
            elapsed = current_time - last_log_time
            average_train_loss = running_loss / steps_since_log
            tokens_processed = (
                steps_since_log * args.batch_size * args.context_length
            )
            tokens_per_second = tokens_processed / max(elapsed, 1e-12)
            tokens_seen = completed_step * args.batch_size * args.context_length

            logger.log(
                step=completed_step,
                split="train",
                loss=average_train_loss,
                lr=learning_rate,
                tokens_per_second=tokens_per_second,
                tokens_seen=tokens_seen,
            )
            print(
                f"step={completed_step:>7d} | "
                f"train_loss={average_train_loss:.4f} | "
                f"lr={learning_rate:.3e} | "
                f"tokens/s={tokens_per_second:,.0f}"
            )

            running_loss = 0.0
            steps_since_log = 0
            last_log_time = current_time

        if completed_step % args.eval_interval == 0:
            validation_loss = evaluate_loss(
                model=model,
                dataset=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=device,
                seed=args.seed + 1,
            )
            logger.log(
                step=completed_step,
                split="validation",
                loss=validation_loss,
            )
            print(
                f"step={completed_step:>7d} | val_loss={validation_loss:.4f}"
            )

        if completed_step % args.checkpoint_interval == 0:
            checkpoint_path = (
                logger.checkpoint_dir / f"step_{completed_step:07d}.pt"
            )
            save_checkpoint(
                model=model,
                optimizer=optim,
                iteration=completed_step,
                out=checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # Preserve the last partial logging window when max_steps is not a multiple
    # of log_interval.
    if steps_since_log > 0:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current_time = time.perf_counter()
        elapsed = current_time - last_log_time
        average_train_loss = running_loss / steps_since_log
        tokens_processed = steps_since_log * args.batch_size * args.context_length
        tokens_per_second = tokens_processed / max(elapsed, 1e-12)
        tokens_seen = args.max_steps * args.batch_size * args.context_length

        logger.log(
            step=args.max_steps,
            split="train",
            loss=average_train_loss,
            lr=last_learning_rate,
            tokens_per_second=tokens_per_second,
            tokens_seen=tokens_seen,
        )
        print(
            f"step={args.max_steps:>7d} | "
            f"train_loss={average_train_loss:.4f} | "
            f"lr={last_learning_rate:.3e} | "
            f"tokens/s={tokens_per_second:,.0f}"
        )

    final_checkpoint = logger.checkpoint_dir / f"step_{args.max_steps:07d}.pt"
    save_checkpoint(
        model=model,
        optimizer=optim,
        iteration=args.max_steps,
        out=final_checkpoint,
    )

    print("Training finished.")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()