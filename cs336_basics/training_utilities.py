import math
import torch
import numpy as np
import os
import typing
from pathlib import Path

def learning_rate_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_end_step: int,
    ):
    if step < warmup_steps:
        return step*max_lr/warmup_steps
    
    elif step > decay_end_step:
        return min_lr

    else:
        decay_ratio = (step - warmup_steps) / (
        decay_end_step - warmup_steps
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + (max_lr - min_lr) * cosine_factor


def gradient_clipping(
    parameters: [torch.nn.Parameter], 
    max_l2_norm: float
    ) -> None:
    eps = 1e-6

    parameters = [
        param for param in parameters
        if param.grad is not None
    ]

    if not parameters:
        return

    total_norm = torch.sqrt(
        sum(torch.sum(param.grad.detach() ** 2) for param in parameters)
    )

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        with torch.no_grad():
            for param in parameters:
                param.grad.mul_(scale)


def data_loading(
    dataset, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    start = torch.randint(
        low = 0,
        high = len(dataset) - context_length-1,
        size = (batch_size,)
    )

    inputs = torch.stack(
        [
            torch.tensor(
                dataset[starting_point: starting_point+context_length]
            )
            for starting_point in start.tolist()
        ]
    ).to(device)
    
    targets = torch.stack(
        [
            torch.tensor(
            dataset[starting_point+1: starting_point+context_length+1]
        )
        for starting_point in start.tolist()
        ]        
    ).to(device)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,  
    optimizer: torch.optim.Optimizer,  
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    )-> None:

    checkpoint={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],  
    model: torch.nn.Module,  
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]


def load_token_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Token file does not exist: {path}")

    data = np.load(path, mmap_mode="r")

    if data.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional token array, got shape {data.shape}"
        )

    return data


