import torch.nn as nn


class ChebyshevLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        cheb = (input-target).abs().max(dim=1)[0]
        cheb = cheb.mean()
        return cheb
