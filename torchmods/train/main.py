import torch
from tqdm import tqdm

from .utils import check_loss, get_loaders, load_checkpoint, save_checkpoint


# Sample transform
# sample_transform = A.Compose([
#     A.Resize(224),
#     A.RandomRotation(30),
#     A.RandomCrop((224, 224)),
#     A.RandomHorizontalFlip(p=0.5),
#     A.ToTensor(),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# Hyper parameters
NUM_WORKERS = 2
PIN_MEMORY = True
MAX_EPOCH = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOOTH_FACTOR = 0.2


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    smooth_loss = None

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
        loss = loss.item()
        if smooth_loss is None:
            smooth_loss = loss
        smooth_loss = SMOOTH_FACTOR * smooth_loss + (1-SMOOTH_FACTOR) * loss
        loop.set_postfix(loss=loss, smooth_loss=smooth_loss)


def loop_fn(model, optimizer, loss_fn, trainset, valset, batch_size, max_patience=10, checkpoint=None, model_name='my_model'):
    if checkpoint is not None:
        load_checkpoint(model, optimizer, checkpoint)

    train_loader, val_loader = get_loaders(trainset, valset, batch_size, NUM_WORKERS, PIN_MEMORY)
    min_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(MAX_EPOCH):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        loss = check_loss(val_loader, model, loss_fn, device=DEVICE)
        # early stop
        if loss < min_loss:
            save_checkpoint(model, optimizer, file_name=f'{model_name}.pth.tar')
            min_loss = loss
            patience = max_patience
        else:
            patience -= 1
            if patience == 0:
                break
