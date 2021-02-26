import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        if len(target.shape) > 1:
            target = target.argmax(dim=1)
        _, input = input.max(dim=1)
        acc = input.eq(target).sum().float()/target.size(0)
        return acc
