from collections.abc import Callable, Iterable
from typing import Optional

import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss



def train():
    lr = [1, 1e1, 1e2, 1e3]
    for l in lr:
        print(f"\nLearning rate: {l}")

        torch.manual_seed(0)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

        opt = SGD([weights], l)

        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.

            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())

            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.



if __name__ == "__main__":
    train()


'''
    Learning rate: 1
    26.271400451660156
    25.231054306030273
    24.5224609375
    23.959409713745117
    23.482616424560547
    23.064424514770508
    22.689321517944336
    22.34758758544922
    22.032663345336914
    21.7398738861084

    Learning rate: 10.0
    26.271400451660156
    16.813697814941406
    12.394339561462402
    9.697248458862305
    7.854771614074707
    6.512506008148193
    5.492434024810791
    4.693441390991211
    4.053155899047852
    3.5307493209838867

    Learning rate: 100.0(best)
    26.271400451660156
    26.271400451660156
    4.507460117340088
    0.10787365585565567
    1.1026770860307538e-16
    1.2290021032668667e-18
    4.1384821860656846e-20
    2.4653223525767597e-21
    2.114912736614738e-22
    2.3499031984552234e-23

    Learning rate: 1000.0
    26.271400451660156
    9483.9765625
    1638032.0
    182213552.0
    14759298048.0
    931480731648.0
    47819176017920.0
    2057385568894976.0
    7.583084306654822e+16
    2.4350125357331907e+18
'''