import torch
from torch.utils.data import DataLoader


def save_checkpoint(model, optimizer, file_name='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        file_name,
    )


def load_checkpoint(model, optimizer, file_name='my_checkpoint.pth.tar'):
    print("=> Loading checkpoint")
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def check_loss(dataloader, model, loss_fn, device):
    model.eval()
    loss = 0

    for data, targets in dataloader:
        data = data.to(device=device)
        targets = targets.to(device=device)

        with torch.no_grad():
            predictions = model(data)
            loss += loss_fn(predictions, targets).item()

    print(
        f'Average {loss_fn} over {len(dataloader)} batches: {loss/len(dataloader):.4f}'
    )
    model.train()
    return loss


def get_loaders(dataset_cls, dataset_root, batch_size, train_transform, val_transform, num_workers, pin_memory):

    trainset = dataset_cls(
        root=dataset_root,
        train=True,
        transforms=train_transform
    )

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valset = dataset_cls(
        root=dataset_root,
        train=False,
        transforms=val_transform
    )

    valloader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, valloader
