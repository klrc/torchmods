import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.transforms as A
from tqdm import tqdm

from .utils import check_loss, get_loaders, load_checkpoint, save_checkpoint


class Parameters():
    # Hyper parameters
    train_transform = A.Compose(
        [
            A.RandomHorizontalFlip(p=0.5),
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = A.Compose(
        [
            A.ToTensor(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_cls = torchvision.datasets.CIFAR10
    dataset_root = '.'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_id = None

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    batch_size = 64
    patience = 8
    max_epoch = 50
    num_workers = 2
    pin_memory = True

    checkpoint_name = 'my_checkpoint'


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main(model, checkpoint=None):

    if Parameters.device_id is not None:
        torch.cuda.set_device(Parameters.device_id)

    model = model.to(device=Parameters.device)
    optimizer = optim.Adam(model.parameters(), lr=Parameters.learning_rate)

    if checkpoint is not None:
        load_checkpoint(model, optimizer, checkpoint)

    train_loader, val_loader = get_loaders(
        Parameters.dataset_cls,
        Parameters.dataset_root,
        Parameters.batch_size,
        Parameters.train_transform,
        Parameters.val_transform,
        Parameters.num_workers,
        Parameters.pin_memory,
    )

    scaler = torch.cuda.amp.GradScaler()
    min_loss = float('inf')
    patience = Parameters.patience

    for epoch in range(Parameters.max_epoch):
        train_fn(train_loader, model, optimizer, Parameters.loss_fn, scaler)

        # check kldiv
        loss = check_loss(val_loader, model, Parameters.loss_fn, device=Parameters.device)

        # early stop
        if loss < min_loss:
            save_checkpoint(model, optimizer, file_name=f'{Parameters.checkpoint_name}.pth.tar')
            min_loss = loss
            patience = Parameters.patience
        else:
            patience -= 1
            if patience == 0:
                break

    del model
    del optimizer
