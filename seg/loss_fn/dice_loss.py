import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.smooth = cfg['smooth']

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = inp.size()[1]

        # Make target size (B, 1, H, W) -> (B, H, W) then use one_hot -> (B, H, W, num_classes)
        # then use view -> (B, num_classes, H, W)
        target = target.squeeze(1).long()
        target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        
        iflat = inp.contiguous().view(-1)
        tflat = target.contiguous().view(-1)

        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))
