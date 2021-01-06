from typing import Union, Tuple
import os
import imageio
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


class NumpyImage():
    def __init__(self, x, imtype=np.uint8):
        self.data = x.astype(imtype)

    def save(self, output):
        plt.imshow(self.data)
        plt.savefig(output)
        plt.close()
        return self

    def __add__(self, other):
        if isinstance(other, NumpyImage):
            other = other.data
        return NumpyImage(self.data + other)

    def __mul__(self, other):
        if isinstance(other, NumpyImage):
            other = other.data
        return NumpyImage(self.data * other)


def denormalize(x):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not isinstance(x, np.ndarray):
        if isinstance(x, torch.Tensor):  # get the data from a variable
            if x.shape[0] == 1 and len(x.shape) == 4:
                x = x[0]
            image_tensor = x.data
        else:
            return x
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy[image_numpy < 0] = 0
        image_numpy[image_numpy > 1] = 1
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = x
    return NumpyImage(image_numpy)


def heatmap(x):
    if len(x.shape) == 4:
        x = x[0]
    assert x.shape[0] == 1
    heatmap = x[0].cpu().data.numpy()
    heatmap = heatmap/np.max(heatmap)
    # must convert to type unit8
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return NumpyImage(heatmap)


class Gif():
    def __init__(self, dir, sort_func=lambda x: x, reverse=False):
        self.data = self.load(dir, sort_func, reverse)

    @staticmethod
    def load(dir, sort_func=lambda x: x, reverse=False):
        files = [x for x in os.listdir(dir) if x.endswith('jpg') or x.endswith('png')]
        files.sort(key=sort_func, reverse=reverse)
        imgs = [imageio.imread(f'{dir}/{img_file}') for img_file in files]
        return imgs

    def save(self, output, fps=8):
        imageio.mimsave(output, self.data, fps=fps)
        return self


class ImageGrid():
    def __init__(self, shape: Union[int, Tuple[int, int]]):
        self.data = []
        if type(shape) is not Tuple:
            shape = (1, shape)
        self.shape = shape
        self._maxlen = shape[0]*shape[1]

    def push(self, x):
        if len(self.data) >= self._maxlen:
            raise OverflowError
        if isinstance(x, NumpyImage):
            x = x.data
        self.data.append(x)
        return self

    def save(self, output, axis_off=True, title=None, quality=4):
        # reshape data
        lines, rows = self.shape
        plt.figure(figsize=(rows*quality, lines*quality))
        for i, x in enumerate(self.data):
            if axis_off:
                plt.axis('off')
            subp = plt.subplot(lines, rows, i+1)
            if title:
                subp.set_title(title)
            plt.imshow(x)
        if axis_off:
            plt.axis('off')
        plt.savefig(output)
        plt.close()
        return self

    def clear(self):
        self.data = []
        return self
