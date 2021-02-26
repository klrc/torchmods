import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class FlickrDataset(Dataset):

    def __init__(self, root, train=True, transforms=None):
        super().__init__()

        # read labels
        with open(root + '/ground_truth.txt', 'r') as f:
            lines = [x.replace('\n', '') for x in f.readlines()]

        self.label_title = lines.pop(0).split(' ')
        self.labels = {}
        for line in lines:
            line = line.split(' ')
            self.labels[line[0]] = line[1:]

        # read samples
        self.__root = os.path.join(root, 'train' if train else 'test')
        self.images = os.listdir(self.__root)

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.__root, image_name)
        image = Image.open(image_path).convert('RGB')
        label = np.array(self.labels[image_name], dtype=np.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        label = label / sum(label)

        return image, label
