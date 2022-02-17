import torch
import torch.nn as nn

from .crossentropy_loss import CrossEntropyLoss
from .dice_loss import DiceLoss


class Criterion(nn.Module):
    def __init__(self, loss_fns_weight_dict: dict, loss_fns_cfg: dict):
        super().__init__()

        self.loss_fns = {}
        self.represent = ''

        for loss_fn_name, weight in loss_fns_weight_dict.items():
            loss_fn = self._build_loss_fn(loss_fn_name, loss_fns_cfg[loss_fn_name])

            self.loss_fns[loss_fn] = weight

            self.represent += f' ,{loss_fn_name}: {weight}'

        self.represent = self.represent[2:]

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = 0

        for loss_fn, weight in self.loss_fns.items():
            loss = loss_fn(inp, target) * weight
            losses += loss

        return losses

    def _build_loss_fn(self, loss_fn_name: str, loss_fn_cfg: dict) -> nn.Module:
        return eval(f'{loss_fn_name}(loss_fn_cfg)')

    def __str__(self) -> str:
        return self.represent


def build_criterion_helper(loss_fns_weighted_dict: dict, loss_fns_cfg: dict) -> Criterion:
    return Criterion(loss_fns_weighted_dict, loss_fns_cfg)
