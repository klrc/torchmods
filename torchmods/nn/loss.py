
import torch
from torch import nn as nn


class Accuracy(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        if len(target.shape) > 1:
            target = target.argmax(dim=1)
        _, input = input.max(dim=1)
        acc = input.eq(target).sum().float()/target.size(0)
        return acc


class CorrelationCoefficientsLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

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


class ChebyshevLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        cheb = (input-target).abs().max(dim=1)[0]
        cheb = cheb.mean()
        return cheb


class ClarkLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        clark = (input-target)/(input+target)
        clark = torch.norm(clark, dim=1).mean()
        return clark


class CanberLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        canber = (input-target).abs()/(input+target)
        canber = canber.sum(dim=1).mean()
        return canber


class CosineLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        cosine = nn.functional.cosine_similarity(input, target, dim=1).mean()
        return cosine


class IntersecLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super().__init__()
        if reduction != 'batchmean':
            raise NotImplementedError

    def forward(self, input, target):
        intersec = input.min(target).sum(dim=1).mean()
        return intersec


class SoftmaxWrapper(nn.Module):
    def __init__(self, core, softmax_input=True, softmax_target=False):
        super().__init__()
        self.core = core
        self.softmax_input = softmax_input
        self.softmax_target = softmax_target

    def adaptive_softmax(self, x):
        if len(x.shape) > 2:
            # some issue with torch.nn.softmax2d so we use this.
            shape = x.shape
            x = x.flatten(start_dim=2)
            x = x.softmax(dim=2)
            x = x.view(shape)
        else:
            x = x.softmax(dim=1)
        return x

    def forward(self, src, target):
        if self.softmax_input:
            src = self.adaptive_softmax(src)
        if self.softmax_target:
            target = self.adaptive_softmax(target)
        return self.core(src, target)


class LogSoftmaxWrapper(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def adaptive_softmax(self, x):
        if len(x.shape) > 2:
            # some issue with torch.nn.softmax2d so we use this.
            shape = x.shape
            x = x.flatten(start_dim=2)
            x = x.log_softmax(dim=2)
            x = x.view(shape)
        else:
            x = x.log_softmax(dim=1)
        return x

    def forward(self, src, target):
        src = self.adaptive_softmax(src)
        return self.core(src, target)


class LogWrapper(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, src, target):
        return self.core(torch.log(src), target)


class Softmax2d_(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        x = x.flatten(start_dim=2)
        x = x.softmax(dim=2)
        x = x.view(shape)
        return x


if __name__ == "__main__":
    loss = SoftmaxWrapper(CorrelationCoefficientsLoss()).cuda()
    x = torch.rand(4, 1, 30, 40).cuda()
    target = torch.rand(4, 1, 30, 40).cuda()
    print(loss(x, target))
