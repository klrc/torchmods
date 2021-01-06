import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchmods.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchmods.nn.atomic import inference, step
from torchmods.utils import find_path, printlog

from .dataset_collection import dataloaders, datasets


def __maxidx__(_list):
    return _list.index(max(_list))


class LDL(data.Dataset):
    img_dir = 'images'
    seed = 0

    def __init__(self, root, train=True, transform=None, lb_type='dist'):
        super().__init__()
        self.root = root
        self.transform = transform
        self.lb_type = lb_type
        self.load_data(root, train)
        assert lb_type in ['dist', 'onehot']
        if lb_type == 'dist':
            self.dist()
        elif lb_type == 'onehot':
            self.onehot()

    def __gensplit__(self, root):
        """
            generate train/test split as split.json,
            Args:
                root:
        """
        np.random.seed(self.seed)
        _list = [x for x in os.listdir(f'{root}/{self.img_dir}')]
        random.shuffle(_list)
        pivod = int(len(_list)*0.2)
        _dict = {'train': _list[pivod:], 'test': _list[:pivod]}
        with open(f'{root}/split.json', 'w') as f:
            json.dump(_dict, f)

    def __readgt__(self, root, spliter=' ', start_line=1, suffix=''):
        """
            read groundtruth from ground_truth.txt,
            (override if more complex func needed),
            Args:
                root:
                spliter: spliter for labels
                start_line: line to start from
                suffix: image file suffix
        """
        _dict = {}
        with open(f'{root}/ground_truth.txt', 'r') as f:
            for line in f.readlines()[start_line:]:  # images-name Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
                data = line.split(spliter)
                _dict[f'{data[0]}{suffix}'] = data
        return [_dict[x.split('/')[-1]] for x in self.data]

    # def __cvimg__(self, path, convert2rgb=True):
    #     img_cv = cv2.imread(path)
    #     if convert2rgb:
    #         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    #     return img_cv

    def __getitem__(self, index):
        # deprecated method (because of thread lock issue:https://zhuanlan.zhihu.com/p/133707658)
        # img = self.__cvimg__(self.data[index])
        img = Image.open(self.data[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

    def load_data(self, root, train):
        assert os.path.exists(root)
        _path = f'{root}/split.json'
        if not os.path.exists(_path):
            self.__gensplit__(root)
        with open(_path, 'r') as f:
            _json = json.load(f)
            self.data = [f'{root}/{self.img_dir}/{x}' for x in _json['train' if train else 'test']]

    def load_label(self, root):
        for x in self.__readgt__(root):
            y = [int(_str.replace('\n', '')) for _str in x[1:]]
            if self.lb_type == 'onehot':
                y = __maxidx__(y)
                y = torch.tensor(y)
            elif self.lb_type == 'dist':
                y = (np.array(y)/sum(y)).astype(np.float32)
                y = torch.from_numpy(y)
            self.labels.append(y)

    def dist(self):
        self.lb_type == 'dist'
        self.labels = []
        self.load_label(self.root)

    def onehot(self):
        self.lb_type == 'onehot'
        self.labels = []
        self.load_label(self.root)


class EnvLDL():
    def __init__(self):
        self.root = find_path(['/home/sh/datasets/Flickr_LDL', '/home/public/datasets/Flickr_LDL'])
        self.transforms = (
            transforms.Compose([
                transforms.Resize(size=(240, 320)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(size=(240, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(size=(240, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        self.max_training_patience = 8
        self.checkpoints = torch.linspace(0, 1, steps=20)
        self.max_epoch = 1000
        self.batch_size = 32
        self.num_workers = 2
        self.pin_memory = False

    def sample(self, batch_size=1, shuffle=True):
        from torch.utils.data import DataLoader
        for batch in DataLoader(LDL(root=self.root, train=False, transform=self.transforms[-1]), batch_size=batch_size, shuffle=shuffle):
            return batch

    def check(self, model, optimizer):
        torch.autograd.set_detect_anomaly(True)
        self.train(model, optimizer, _test_mode=True)
        self.performance(model, _test_mode=True)
        print('> passed.')

    def train(self, model, optimizer, _test_mode=False):
        trainloader, valloader, testloader = dataloaders(
            datasets(LDL, root=self.root, transforms=self.transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        min_average_kldiv_test = float('inf')
        patience = self.max_training_patience
        criterion = nn.LogSoftmaxWrapper(nn.KLDivLoss(reduction='batchmean'))
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

    def performance(self, model,  verbose=False, _test_mode=False):
        testloader = torch.utils.data.DataLoader(
            LDL(root=self.root, train=False, transform=self.transforms[-1]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        # test performance
        model.eval()
        softmax = nn.Softmax(dim=1)
        metrics_patch = {
            'Acc': [nn.Accuracy(), []],
            'Cheb': [nn.ChebyshevLoss(), []],
            'Clark': [nn.ClarkLoss(), []],
            'Canber': [nn.CanberLoss(), []],
            'KLDiv': [nn.LogWrapper(nn.KLDivLoss(reduction='batchmean')), []],
            'Cosine': [nn.CosineLoss(), []],
            'Intersec': [nn.IntersecLoss(), []],
        }
        with torch.no_grad():
            for iteration, (x, target) in enumerate(testloader):
                x, target = x.cuda(), target.cuda()
                y = softmax(model(x))
                logs = {'iter': iteration, 'mode': 'test'}
                for m in metrics_patch:
                    metrics_patch[m][1].append(metrics_patch[m][0](y, target).item())
                    v = sum(metrics_patch[m][1])/len(metrics_patch[m][1])
                    logs[m] = v
                if verbose:
                    printlog(**logs)
                if _test_mode:
                    break
            printlog(**logs)
        return metrics_patch
