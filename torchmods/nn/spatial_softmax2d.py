
import torch.nn as nn


class SpatialSoftmax2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        x = x.flatten(start_dim=2)
        x = x.softmax(dim=2)
        x = x.view(shape)
        return x
