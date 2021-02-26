
import torch.nn as nn


class InputMask(nn.Module):
    def __init__(self, module, module_pred=None, module_target=None):
        super().__init__()
        self.module_inner = module
        self.module_pred = module_pred
        self.module_target = module_target

    def forward(self, inputs, targets):
        arg1 = inputs if self.module_pred is None else self.module_pred(inputs)
        arg2 = targets if self.module_target is None else self.module_target(targets)
        return self.module_inner(arg1, arg2)
