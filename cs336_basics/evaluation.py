from __future__ import annotations

import numpy as np
import torch

from cs336_basics.layers import cross_entropy
from cs336_basics.training_utilities import get_batch


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str | torch.device,
    seed: int = 0,
) -> float:
    if eval_iters <= 0:
        raise ValueError("eval_iters must be positive")

    was_training = model.training
    model.eval()

    generator = torch.Generator()
    generator.manual_seed(seed)

    losses: list[float] = []

    try:
        for _ in range(eval_iters):
            inputs, targets = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            losses.append(loss.item())
    finally:
        model.train(was_training)

    return sum(losses) / len(losses)