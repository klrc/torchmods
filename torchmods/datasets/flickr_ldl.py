import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import json


class FlickrDataset(Dataset):

    def __init__(self, root, train=True, transforms=None, fold=0):
        super().__init__()

        with open(root + f'/checklist_fold_{fold}.json', 'r') as f:
            self.data = json.load(f)['train' if train else 'val']

        # read samples
        self.root = os.path.join(root, 'data/')
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path).convert('RGB')
        label = np.array(label, dtype=np.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        label = label / sum(label)

        return image, label


if __name__ == "__main__":
    model = FlickrDataset('/home/sh/Datasets/Flickr_LDL')
    print(model.__getitem__(1))
