import os
import shutil
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
from typing import Iterator, Tuple
from PIL import Image
import cv2

from .dataset import build_dataloader_helper
from .transform import build_transform_helper, DataTransform
from .model import build_model_helper
from .loss_fn import Criterion, build_criterion_helper
from .utils import get_logger, AverageMeter, TrainTimeHelper, combine_submissions


logger = get_logger(__name__)


class SegSolver(object):
    def __init__(self, cfg: dict):
        self.cwd = os.getcwd()
        self.cfg = cfg

    def _setup_env(self, cfg: dict) -> None:
        logger.info(f'Setting up the environment......................\n')

        expr_cfg = cfg['expr']
        train_cfg = cfg['train']
        models_cfg = cfg['models']
        datasets_cfg = cfg['datasets']
        optimizers_cfg = cfg['optimizers']
        schedulers_cfg = cfg['schedulers']
        loss_fns_cfg = cfg['loss_fns']
        transforms_cfg = cfg['transforms']

        self.is_cuda = train_cfg['is_cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.is_cuda else torch.device('cpu')

        random_seed = expr_cfg['random_seed']
        torch.manual_seed(random_seed)

        model_name = expr_cfg['model']
        dataset_name = expr_cfg['dataset']
        optimizer_name = expr_cfg['optimizer']
        scheduler_name = expr_cfg['scheduler']
        loss_fns_weighted_dict = expr_cfg['loss_fn']

        batch_size = train_cfg['batch_size']

        self.model = self._build_model(model_name, models_cfg[model_name])
        self.transform = self._build_transform(transforms_cfg)
        self.train_dataloader, self.val_dataloader = \
            self._build_dataloader(dataset_name, datasets_cfg[dataset_name], batch_size, self.transform)
        self.optimizer = self._build_optimizer(optimizer_name, self.model.parameters(), 
                                               optimizers_cfg[optimizer_name])
        self.scheduler = self._build_scheduler(scheduler_name, self.optimizer, 
                                               schedulers_cfg[scheduler_name])
        self.criterion = self._build_criterion(loss_fns_weighted_dict, loss_fns_cfg)

        self.best_val_loss = 0
        self.best_val_epoch = 0

        data_name = datasets_cfg[dataset_name]['data_dir']
        logger.info(f'The configuration of this experiment is as follows:')
        logger.info(f'The model is: {model_name}.')
        logger.info(f'The dataset is: {dataset_name}, data_dir is {data_name}.')
        logger.info(f'The optimizer is: {optimizer_name}.')
        logger.info(f'The criterion is: {str(self.criterion)}.')

    def run(self) -> None:
        self._setup_env(self.cfg)
        self.train(self.cfg['train'])
        
        test_img_dir = self._get_test_img_dir()
        
        # Generate submission file of model of after train.
        model = os.path.join(self.cwd, 'model.pth')
        submission_file_after_train = os.path.join(self.cwd, 'submission_after_train.csv')
        self.submission(test_img_dir, model, submission_file_after_train)

        # Generate submission file of model of best val.
        model = os.path.join(self.cwd, f'best_epoch{self.best_val_epoch}.pth')
        submission_file_best_val = os.path.join(self.cwd, 'submission_best_val.csv')
        self.submission(test_img_dir, model, submission_file_best_val)

    def _build_transform(self, transforms_cfg: dict) -> DataTransform:
        return build_transform_helper(transforms_cfg)

    def _build_model(self, model_name: str, model_cfg: dict) -> torch.nn.Module:
        return build_model_helper(model_name, model_cfg)

    def _build_dataloader(self, dataset_name: str, dataset_cfg: dict, batch_size: int, 
                          transform: DataTransform, shuffle=True) -> Tuple[DataLoader, DataLoader]:
        return build_dataloader_helper(dataset_name, dataset_cfg, batch_size, transform)

    def _build_optimizer(self, optimizer_name: str, model_params: Iterator[torch.Tensor], 
                         optimizer_cfg: dict) -> torch.optim.Optimizer:
        return eval(f'torch.optim.{optimizer_name}(model_params, **optimizer_cfg)')

    def _build_scheduler(self, scheduler_name: str, optimizer: torch.optim.Optimizer, 
                         scheduler_cfg: dict) -> torch.optim.lr_scheduler._LRScheduler:
        return eval(f'torch.optim.lr_scheduler.{scheduler_name}(optimizer, **scheduler_cfg)')

    def _build_criterion(self, loss_fns_weighted_dict: dict, loss_fns_cfg: dict) -> Criterion:
        return build_criterion_helper(loss_fns_weighted_dict, loss_fns_cfg)

    def get_total_iters(self) -> int:
        return len(self.train_dataloader)

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        img = batch[0].to(device=self.device, dtype=torch.float32)
        mask = batch[1].to(device=self.device, dtype=torch.float32)

        out = self.model(img)
        loss = self.criterion(out, mask)

        return loss

    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, train_cfg: dict) -> None:
        logger.info(f'Start training....................................................\n')
        epochs = int(train_cfg['epochs'])
        total_iters = self.get_total_iters()
        print_freq = int(train_cfg['print_freq'] * total_iters)
        val_freq = int(train_cfg['val_freq'] * epochs)

        logger.info(f'The configuration of training is as follows:')
        logger.info(f'The epochs are {epochs}, total iters are {total_iters}.')
        logger.info(f'The print frequence is {print_freq}.')
        logger.info(f'The evaluate frequence is {val_freq}.')
        logger.info(f'The Cuda use is {self.is_cuda}.')

        loss_meter = AverageMeter()
        traintime_helper = TrainTimeHelper(epochs, total_iters)
        
        self.model = self.model.to(device=self.device)
        self.model.train()
        
        for epoch in range(epochs):
            for iter, batch in enumerate(self.train_dataloader):
                loss = self.forward(batch)
                self.backward(loss)

                loss_meter.update(loss.item())

                if (iter + 1) % print_freq == 0:
                    logger.info(f'Epoch: {epoch + 1}/{epochs}, Iter: {iter + 1}/{total_iters}, \
                                  Loss: {loss_meter.get_avg()}, \
                                  Rest time: {traintime_helper.rest_time(epoch + 1, iter + 1)}.')

            self.scheduler.step()

            if (epoch + 1) % val_freq == 0:
                self.evaluate(epoch + 1)
                self.model.train()
        
        loss = self.evaluate('after train')
        model_saved_path = os.path.join(self.cwd, 'model.pth')
        self.save(model_saved_path)

        logger.info(f'End training...........................................................\n')

        logger.info(f'Train Summary:')
        logger.info(f'The best model is on epoch: {self.best_val_epoch}, loss is: {self.best_val_loss}, saved to {self._get_best_model_path()}.')
        logger.info(f'The model after training is on epoch: {epochs}, loss is: {loss}, saved to {model_saved_path}.')

    def evaluate(self, epoch) -> float:
        self.model.eval()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for _, batch in enumerate(self.val_dataloader):
                img = batch[0].to(device=self.device, dtype=torch.float32)
                mask_target = batch[1].to(device=self.device, dtype=torch.float32)
                out = self.model(img)
                mask_pred = self._tensor_to_mask(out)

                loss = self._dice_loss(mask_pred, mask_target)
                val_loss_meter.update(loss)

        val_loss = val_loss_meter.get_avg()

        logger.info(f'The val loss evaluated on epoch {epoch} is {val_loss}.')
        
        if val_loss > self.best_val_loss:
            logger.info(f'The val loss evaluated on epoch {epoch} is now the best.')

            last_best_model = self._get_best_model_path()
            if os.path.exists(last_best_model):
                os.remove(last_best_model)

            self.best_val_loss = val_loss
            self.best_val_epoch = epoch

            self.save(os.path.join(self.cwd, f'best_epoch{epoch}.pth'))

        return val_loss

    def _get_best_model_path(self):
        return os.path.join(self.cwd, f'best_epoch{self.best_val_epoch}.pth')

    def save(self, path: str) -> None:
        logger.info(f'Save model to {path}.')
        torch.save(self.model.state_dict(), path)

    def _get_test_img_dir(self):
        datasets_cfg = self.cfg['datasets']
        dataset_name = self.cfg['expr']['dataset']
        data_dir = datasets_cfg[dataset_name]['data_dir']

        return os.path.join(self.cwd, data_dir, 'test/image')

    def _tensor_to_mask(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.permute(0, 2, 3, 1)
        # The index of max value of each pixel is exactly the class of the pixel. 
        mask = torch.argmax(inp, dim=3)

        return mask

    # This dice loss can only be used for evluating and can't be used for training.
    def _dice_loss(self, mask_pred: torch.Tensor, mask_target: torch.Tensor) -> float:
        num_classes = 4
        losses = 0

        for class_num in range(num_classes):
            loss = self._categorical_dice(mask_pred, mask_target, class_num)
            losses += loss

        return losses / num_classes
    
    def _categorical_dice(self, mask1: torch.Tensor, mask2: torch.Tensor, label_class: int) -> float:
        mask1_pos = (mask1 == label_class).cpu().numpy().astype(np.float32)
        mask2_pos = (mask2 == label_class).cpu().numpy().astype(np.float32)

        dice = 2 * np.sum(mask1_pos * mask2_pos) / (np.sum(mask1_pos) + np.sum(mask2_pos))

        return dice

    def _rle_encoding(self, x):
        '''
        *** Credit to https://www.kaggle.com/rakhlin/fast-run-length-encoding-python ***
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        '''
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def _submission_converter(self, mask_directory: str, submission_file: str):
        writer = open(submission_file, 'w')
        writer.write('id,encoding\n')

        files = os.listdir(mask_directory)

        for file in files:
            name = file[:-4]
            mask = cv2.imread(os.path.join(mask_directory, file), cv2.IMREAD_UNCHANGED)

            mask1 = (mask == 1)
            mask2 = (mask == 2)
            mask3 = (mask == 3)

            encoded_mask1 = self._rle_encoding(mask1)
            encoded_mask1 = ' '.join(str(e) for e in encoded_mask1)
            encoded_mask2 = self._rle_encoding(mask2)
            encoded_mask2 = ' '.join(str(e) for e in encoded_mask2)
            encoded_mask3 = self._rle_encoding(mask3)
            encoded_mask3 = ' '.join(str(e) for e in encoded_mask3)

            writer.write(name + '1,' + encoded_mask1 + "\n")
            writer.write(name + '2,' + encoded_mask2 + "\n")
            writer.write(name + '3,' + encoded_mask3 + "\n")

        writer.close()
        logger.info(f'Save submission file to {submission_file}.')

    def submission(self, test_img_dir: str, model: str, submission_file: str) -> None:
        logger.info(f'Generating the submission file.....................................\n')
        
        predict_masks_dir = os.path.join(self.cwd, 'predict_test_masks')
        if os.path.exists(predict_masks_dir):
            shutil.rmtree(predict_masks_dir)

        self._generate_test_masks(predict_masks_dir, test_img_dir, model)
        self._submission_converter(predict_masks_dir, submission_file)

    def _generate_test_masks(self, predict_masks_dir: str, test_img_dir: str, model: str) -> None:
        test_imgs = os.listdir(test_img_dir)

        self._load_model(model)
        self.model.eval()

        if not os.path.exists(predict_masks_dir):
            os.mkdir(predict_masks_dir)

        with torch.no_grad():
            for img in test_imgs:
                inp = read_image(os.path.join(test_img_dir, img)).unsqueeze(0).to(device=self.device, dtype=torch.float32)
                mask = self._tensor_to_mask(self.model(inp)).cpu().squeeze(0).numpy().astype(np.int32)

                mask_file_name = img.split('.')[0] + '_mask.png'
                mask_file = os.path.join(predict_masks_dir, mask_file_name)

                mask = Image.fromarray(mask)
                mask.save(mask_file)

    def _load_model(self, model: str) -> None:
        self.model.load_state_dict(torch.load(model))

