import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class SALICONDataset(Dataset):

    def __init__(self, root, train=True, transforms=None):
        super().__init__()
        self.load_data(root, train)
        self.transform = transforms

    def __cvimg__(self, path, convert2rgb=True):
        """
            read image in RGB mode.
        """
        img_cv = cv2.imread(path)
        if convert2rgb:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return img_cv

    def __getitem__(self, index):
        """
            function for torch.util.data.Dataset class,
            returns image, (target_1, target_2, ..., target_n)

            Args:
                index:
        """
        img = self.__cvimg__(self.data[index])
        if self.transform:
            img = self.transform(img)
        target = self.preprocess_maps(self.data[index].replace('/images/', '/maps/').replace('.jpg', '.png'))
        target = torch.from_numpy(target)
        shape = target.shape
        sum_target = torch.nn.functional.adaptive_avg_pool2d(target, output_size=(1, 1))*shape[-2]*shape[-1]
        target /= sum_target
        return img, target

    def __len__(self):
        """
            function for torch.util.data.Dataset class.
        """
        return len(self.data)

    def padding(self, img, shape_r=7, shape_c=7, channels=3):
        img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
        if channels == 1:
            img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

        original_shape = img.shape
        rows_rate = original_shape[0]/shape_r
        cols_rate = original_shape[1]/shape_c

        if rows_rate > cols_rate:
            new_cols = (original_shape[1] * shape_r) // original_shape[0]
            img = cv2.resize(img, (new_cols, shape_r))
            if new_cols > shape_c:
                new_cols = shape_c
            img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
        else:
            new_rows = (original_shape[0] * shape_c) // original_shape[1]
            img = cv2.resize(img, (shape_c, new_rows))
            if new_rows > shape_r:
                new_rows = shape_r
            img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

        return img_padded

    def preprocess_maps(self, path, shape_r=7, shape_c=7):
        ims = np.zeros((1, shape_r, shape_c))

        original_map = cv2.imread(path, 0)
        padded_map = self.padding(original_map, shape_r, shape_c, 1)
        ims[0] = padded_map.astype(np.float32)
        ims[0] /= 255.0

        return ims.astype(np.float32)

    def load_data(self, root, train):
        """
            pre-load image path,
            (override if more complex func needed),
            Args:
                root:
                train:
        """
        assert os.path.exists(root)
        src_dir = 'train' if train else 'val'
        src_dir = f'{root}/images/{src_dir}'
        self.data = [f'{src_dir}/{x}' for x in os.listdir(src_dir)]
