import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False    # Freeze original weights

        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # A random, B zero

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merge(self) -> nn.Linear:
        """Fold B.A into the base weight and return a plain nn.Linear"""
        with torch.no_grad():
            self.base.weight += self.scale * (self.B @ self.A)
        return self.base
