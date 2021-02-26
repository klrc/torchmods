import torch.nn as nn


class IntersecLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        intersec = input.min(target).sum(dim=1).mean()
        return intersec
