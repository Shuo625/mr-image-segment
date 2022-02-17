from typing import Tuple
from torch.utils.data import DataLoader

from .base_dataset import BaseDataset


def build_dataset_helper(dataset_name: str, dataset_cfg: dict, is_train: bool, transform=None) -> BaseDataset:
    return eval(f'{dataset_name}(dataset_cfg, is_train, transform=transform)')


def build_dataloader_helper(dataset_name, dataset_cfg: dict, batch_size, transform=None,
                            shuffle=True) -> Tuple[DataLoader, DataLoader]:
    train_dataset = build_dataset_helper(dataset_name, dataset_cfg, is_train=True, transform=transform)
    val_dataset = build_dataset_helper(dataset_name, dataset_cfg, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader
