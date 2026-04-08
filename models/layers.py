import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = (torch.rand_like(x) > self.p).float()
        mask = mask / (1.0 - self.p)

        return x * mask
