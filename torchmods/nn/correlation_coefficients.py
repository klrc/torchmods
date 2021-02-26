import torch
import torch.nn as nn


class CorrelationCoefficientsLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        assert reduction in ('batchmean',)

    def forward(self, input, target):
        batch_size = input.shape[0]
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        CC = []
        for i in range(batch_size):
            im = input[i] - torch.mean(input[i])
            tm = target[i] - torch.mean(target[i])
            CC.append(torch.sum(im * tm) / (torch.sqrt(torch.sum(im ** 2))
                                            * torch.sqrt(torch.sum(tm ** 2))))
            CC[i].unsqueeze_(0)
        CC = torch.cat(CC, 0)
        CC = torch.mean(CC)
        return CC
