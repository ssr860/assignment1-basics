from collections.abc import Callable, Iterable
from __future__ import annotations

import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight decay: {weight_decay}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历不同的参数组
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历该参数组中的所有参数
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                # 每个parameter独立的优化器state
                state = self.state[parameter]

                # 第一次更新这个参数时初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)

                state["step"] += 1

                step = state["step"]
                m = state["m"]
                v = state["v"]

                adjusted_lr = (
                    lr
                    * math.sqrt(1 - beta2**step)
                    / (1 - beta1**step)
                )

                parameter.add_(
                    parameter,
                    alpha=-lr * weight_decay,
                )

                m.mul_(beta1)
                m.add_(grad, alpha=1 - beta1)

                v.mul_(beta2)
                v.addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                denominator = torch.sqrt(v).add_(eps)

                parameter.addcdiv_(
                    m,
                    denominator,
                    value=-adjusted_lr,
                )

        return loss