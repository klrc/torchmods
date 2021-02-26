
import torch.nn as nn


class CanberLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        canber = (input-target).abs()/(input+target)
        canber = canber.sum(dim=1).mean()
        return canber
