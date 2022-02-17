import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from ..utils import get_logger


logger = get_logger(__name__)


class BaseDataset(Dataset):
    def __init__(self, cfg: dict, is_train: bool, transform=None):
        self.data_dir = self._get_path(cfg['data_dir'])
        self.is_train = is_train
        
        if is_train:
            self.data_dir = os.path.join(self.data_dir, 'train')
        else:
            self.data_dir = os.path.join(self.data_dir, 'val')

        self.img_dir = os.path.join(self.data_dir, 'image')
        self.mask_dir = os.path.join(self.data_dir, 'mask')

        self.img_list = os.listdir(self.img_dir)

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = read_image(img_path).to(dtype=torch.float32)
        
        mask_file_name = self.img_list[idx].split('.')[0] + '_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_file_name)
        mask = read_image(mask_path).to(dtype=torch.float32)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    # Fit the absolute path or relative path
    def _get_path(self, path: str) -> str:
        if path.startswith('/'):
            return path
        else:
            return os.path.join(os.getcwd(), path)
