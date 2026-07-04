import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L1) used for SWIFTFormer optimization."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
