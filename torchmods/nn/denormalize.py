import torch


def denormalize(x):
    mean = torch.tensor([[0.485, 0.456, 0.406]]).to(x.device).view(1, 3, 1, 1)
    std = torch.tensor([[0.229, 0.224, 0.225]]).to(x.device).view(1, 3, 1, 1)
    return x*std + mean
