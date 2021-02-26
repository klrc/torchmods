
import torch
import torch.nn as nn


class ClarkLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        clark = (input-target)/(input+target)
        clark = torch.norm(clark, dim=1).mean()
        return clark
