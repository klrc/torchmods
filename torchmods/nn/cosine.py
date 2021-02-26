import torch.nn as nn


class CosineLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in('batchmean',)

    def forward(self, input, target):
        cosine = nn.functional.cosine_similarity(input, target, dim=1).mean()
        return cosine
