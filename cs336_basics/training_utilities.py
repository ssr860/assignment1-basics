import math
import torch
import numpy
import os
import typing

def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
    ):
    if t < T_w:
        return t*alpha_max/T_w
    
    elif t > T_c:
        return alpha_min

    else:
        return alpha_min+(alpha_max-alpha_min)*(1+math.cos((t-T_w)*math.pi/(T_c-T_w)))/2


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


