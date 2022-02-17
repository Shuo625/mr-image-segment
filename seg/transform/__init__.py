from typing import Tuple
import torch
import torchvision
import random


class WrappedTransform(object):
    def __init__(self, transform_cfg: dict):
        self.transform_type = transform_cfg['type']
        self.need_random_wrap = transform_cfg['need_random_wrap']
        self.is_transform_label = transform_cfg['is_transform_label']
        
        params = transform_cfg['params']
        if self.need_random_wrap:
            self.p = params['p']
            params.pop('p')
        self.transform = eval(f'torchvision.transforms.{self.transform_type}(**params)')

    def _can_run(self) -> True:
        if not self.need_random_wrap:
            return True
        
        return random.random() < self.p
    
    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._can_run():
            return img, mask

        if self.is_transform_label:
            img_cat_mask = torch.cat((img, mask), dim=0)
            return torch.split(self.transform(img_cat_mask), 1)
        else:
            return self.transform(img), mask

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(img, mask)


class DataTransform(object):
    def __init__(self, transforms_cfg: dict):
        self.transforms = []

        for name, transform_cfg in transforms_cfg.items():
            self.transforms.append(WrappedTransform(transform_cfg))

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(img, mask)


def build_transform_helper(transforms_cfg: dict) -> DataTransform:
    if transforms_cfg is None:
        return None
    return DataTransform(transforms_cfg)
