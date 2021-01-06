
import os

import cv2
import numpy as np
import torch
import torchmods.nn as nn
import torchvision.transforms as transforms
from torchmods.nn.atomic import inference, step
from torchmods.utils import find_path, nvidia, printlog

from .dataset_collection import dataloaders, datasets


def padding(img, shape_r=240, shape_c=320, channels=3):
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


def preprocess_maps(path, shape_r=240, shape_c=320):
    ims = np.zeros((1, shape_r, shape_c))

    original_map = cv2.imread(path, 0)
    padded_map = padding(original_map, shape_r, shape_c, 1)
    ims[0] = padded_map.astype(np.float32)
    ims[0] /= 255.0

    return ims.astype(np.float32)


class SALICON(torch.utils.data.Dataset):
    img_dir = 'images'
    seed = 0

    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.load_data(root, train)
        self.transform = transform
        self.lb_type = 'maps'

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
        if self.lb_type == 'maps':
            target = preprocess_maps(self.data[index].replace('/images/', '/maps/').replace('.jpg', '.png'))
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


class EnvSALICON():
    def __init__(self):
        self.root = find_path(['/home/sh/datasets/SALICON', '/home/public/datasets/SALICON'])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.max_training_patience = 8
        self.checkpoints = torch.linspace(0, 1, steps=20)
        self.max_epoch = 100
        self.batch_size = 16
        self.num_workers = 2
        self.pin_memory = True

    def sample(self, batch_size=1, shuffle=True):
        from torch.utils.data import DataLoader
        for batch in DataLoader(SALICON(root=self.root, train=False, transform=self.transform), batch_size=batch_size, shuffle=shuffle):
            return batch

    def check(self, model, optimizer):
        model_size = nvidia.state()['used']
        torch.autograd.set_detect_anomaly(True)
        self.train(model, optimizer, _test_mode=True)
        self.performance(model, _test_mode=True)
        print(f'> model size: {model_size}.')
        print('> passed.')

    def train(self, model, optimizer, _test_mode=False):
        trainloader, valloader, testloader = dataloaders(
            datasets(SALICON, root=self.root, transforms=(self.transform,)*3),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        criterion = nn.LogSoftmaxWrapper(nn.KLDivLoss(reduction='batchmean'))
        min_average_kldiv_test = float('inf')
        patience = self.max_training_patience
        checkpoints = (self.checkpoints*len(trainloader)).numpy().astype('int').tolist()
        printlog(checkpoints=checkpoints)
        for epoch in range(self.max_epoch):
            # train model on full trainset
            model.train()
            loss_patch = []
            for iteration, (x, target) in enumerate(trainloader):
                x, target = x.cuda(), target.cuda()
                loss_patch.append(step(model, x, target, criterion, optimizer))
                if iteration in checkpoints:
                    loss = sum(loss_patch)/len(loss_patch)
                    printlog(epoch=epoch, iter=iteration, loss=loss, mode='train')
                    loss_patch = []
                # if epoch == 0 and iteration in [int(_r*len(trainloader)) for _r in (0.05, 0.1, 0.2, 0.5)]:  # quick check
                #     printlog(epoch=epoch, iter=iteration, checkpoint='quick check')
                #     torch.save(model.state_dict(), f'last.pth')
                if _test_mode:
                    break
            # validation on full valset
            model.eval()
            loss_patch = []
            with torch.no_grad():
                for iteration, (x, target) in enumerate(valloader):
                    x, target = x.cuda(), target.cuda()
                    loss_patch.append(inference(model, x, target, criterion))
                    if _test_mode:
                        break
            loss = sum(loss_patch)/len(loss_patch)
            printlog(epoch=epoch, iter=iteration, loss=loss, mode='validation')
            if _test_mode:
                break
            if loss < min_average_kldiv_test:
                min_average_kldiv_test = loss
                patience = self.max_training_patience
                torch.save(model.state_dict(), f'last.pth')
            else:
                patience -= 1
                if patience < 1:
                    break
        return model

    def performance(self, model, _test_mode=False):
        testloader = torch.utils.data.DataLoader(
            SALICON(root=self.root, train=False, transform=self.transform),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # test performance
        model.eval()
        softmax2d = nn.Softmax2d_()
        metrics_patch = {'CC': [nn.CorrelationCoefficientsLoss(), []]}
        with torch.no_grad():
            for iteration, (x, target) in enumerate(testloader):
                x, target = x.cuda(), target.cuda()
                y = softmax2d(model(x))
                logs = {'iter': iteration, 'mode': 'test'}
                for m in metrics_patch:
                    metrics_patch[m][1].append(metrics_patch[m][0](y, target))
                    v = sum(metrics_patch[m][1])/len(metrics_patch[m][1])
                    logs[m] = v
                printlog(**logs)
                if _test_mode:
                    break
        return metrics_patch
