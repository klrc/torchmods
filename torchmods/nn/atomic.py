import torch


def step(model: torch.nn.Module,
         x: torch.Tensor,
         target: torch.Tensor,
         criterion,
         optimizer: torch.optim.Optimizer,
         **kwargs):
    assert model.training
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, target, **kwargs)
    loss.backward()
    optimizer.step()
    return loss.item()


def inference(model: torch.nn.Module,
              x: torch.Tensor,
              target: torch.Tensor,
              criterion,
              **kwargs):
    assert not model.training
    y = model(x)
    loss = criterion(y, target, **kwargs)
    return loss.item()
