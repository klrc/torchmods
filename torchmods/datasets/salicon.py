import os
from torch.utils.data import Dataset
import cv2


class SALICONDataset(Dataset):

    def __init__(self, root, train=True, transforms=None):
        super().__init__()
        assert os.path.exists(root)
        src_dir = 'train' if train else 'val'
        src_dir = f'{root}/images/{src_dir}'
        self.data = [f'{src_dir}/{x}' for x in os.listdir(src_dir)]
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
        target = cv2.imread((self.data[index].replace('/images/', '/maps/').replace('.jpg', '.png')), 0)

        # for albumentations only!
        if self.transform:
            output = self.transform(image=img, mask=target)
            img = output['image']
            target = output['mask']

        return img, target

    def __len__(self):
        """
            function for torch.util.data.Dataset class.
        """
        return len(self.data)
