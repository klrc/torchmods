import numpy as np
import cv2
import torch


def heatmap_numpy(tensor, colormap=cv2.COLORMAP_JET):
    assert tensor.shape[0] == 1 and tensor.shape[1] == 1
    tensor = (tensor - tensor.min())/(tensor.max() - tensor.min())
    tensor = tensor.view(*tensor.shape[-2:])
    heatmap = cv2.applyColorMap(np.uint8(255 * tensor), colormap)
    return heatmap


def heatmap(tensor, colormap=cv2.COLORMAP_JET):
    ndarr = heatmap_numpy(tensor, colormap)
    tensor = torch.from_numpy(ndarr)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor
