# src/transforms/normalize.py
import torch.nn as nn

class Normalize1D(nn.Module):
    def __init__(self, mean, std):
        super(Normalize1D, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std
