import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.nn_ce_loss = nn.CrossEntropyLoss(**cfg)

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = inp.size()[1]

        # Make (B, C, H, W) -> (B, H, W, C).
        inp = inp.permute(0, 2, 3, 1)
        inp = inp.contiguous()
        target = target.long()
        inp = inp.view(-1, num_classes)
        target = target.view(-1)

        return self.nn_ce_loss(inp, target)
