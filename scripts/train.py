from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import argparse
import math
from pathlib import Path
import time
import csv
import json
import time
from dataclasses import asdict

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.get_tokenizer import Tokenizer
from cs336_basics.train_bpe import run_train_bpe as i_run_train_bpe
from cs336_basics import layers, blocks, optimizer, training_utilities



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train-data-path",
        type=Path,
        required = True,
    )
    parser.add_argument(
        "--val-data-path",
        type = Path,
        required = True,
    )
    parser.add_argument(
        "--checkpoint-path",
        type = Path,
        required = True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/default"),
    )

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-end-step", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Use --device cpu or fix the CUDA environment."
        )

    device = torch.device(args.device)

    print("Loading token arrays with memory mapping.")

    train_data = load_token_array(args.train_data_path)
    val_data = load_token_array(args.val_data_path)

    print(f"Training tokens:   {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Training dtype:    {train_data.dtype}")
    print(f"Device:            {device}")

    model = blocks.Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
        device=device,
    ).to(device)

    optim = optimizer.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0

    if args.resume:
        start_step = load_checkpoint(
            checkpoint_path=args.checkpoint_path,
            model=model,
            optimizer=optim,
            device=device,
        )

        print(
            f"Resumed from {args.checkpoint_path} "
            f"at step {start_step}."
        )

    model.train()
    last_log_time = time.perf_counter()

    for step in range(start_step, args.max_steps):
        learning_rate = training_utilities.learning_rate_schedule(
            step=step,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            decay_end_step=args.decay_end_step,
        )

        for parameter_group in optim.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = training_utilities.get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        optim.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = layers.cross_entropy(logits.size(-1), targets)

        loss.backward()

        training_utilities.gradient_clipping(
            model.parameters(),
            args.max_grad_norm,
        )

        optim.step()

        completed_step = step + 1

        if completed_step % args.log_interval == 0:
            current_time = time.perf_counter()
            elapsed = current_time - last_log_time

            tokens_processed = (
                args.log_interval
                * args.batch_size
                * args.context_length
            )

            tokens_per_second = tokens_processed / elapsed

            print(
                f"step={completed_step:>7d} | "
                f"train_loss={loss.item():.4f} | "
                f"lr={learning_rate:.3e} | "
                f"tokens/s={tokens_per_second:,.0f}"
            )

            last_log_time = current_time

        if completed_step % args.eval_interval == 0:
            val_loss = layers.cross_entropy(
                model=model,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                eval_iters=args.eval_iters,
                device=device,
            )

            print(
                f"step={completed_step:>7d} | "
                f"val_loss={val_loss:.4f}"
            )

        if completed_step % args.checkpoint_interval == 0:
            training_utilities.save_checkpoint(
                checkpoint_path=args.checkpoint_path,
                model=model,
                optimizer=optim,
                step=completed_step,
                args=args,
            )

            print(
                f"Checkpoint saved to {args.checkpoint_path}"
            )

    save_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        optimizer=optim,
        step=args.max_steps,
        args=args,
    )

    print("Training finished.")
    print(f"Final checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()