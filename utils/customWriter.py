"""
Tensorboard writer
"""
import matplotlib
matplotlib.use('Agg')
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch

class customWriter(SummaryWriter):
    def __init__(self, batch_size):
        super().__init__()
        self.metrics = {'train_loss': [], 'val_loss': [], 
                        'BCE': [], 'DSC': []}
        self.epoch = 0
        self.batch_size = batch_size

    def reset_losses(self):
        self.metrics = {key: [] for key in self.metrics.keys()}

    @staticmethod
    def norm_img(img):
        return (img-img.min())/(img.max()-img.min())
    
    @staticmethod
    def sigmoid(x):
        return 1/(1+torch.exp(-x))

    def plot_inputs(self, title, img):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Inputs')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img, cmap='gray')
            ax.set_title(f'Input @ epoch: {self.epoch} - idx: {idx}')
        self.add_figure(title, fig)
        self.flush()

    def plot_segmentation(self, title, img, prediction, targets=None):
        fig = plt.figure(figsize=(10, 10))
        plt.tight_layout()
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(self.batch_size // 2, self.batch_size // 2,
                                 idx+1, label='Predictions')
            plt_img = np.moveaxis(img[idx].cpu().numpy(), 0, -1)
            plt_img = self.norm_img(plt_img)
            ax.imshow(plt_img, cmap='gray')
            pred = self.sigmoid(prediction[idx]).cpu().numpy()[0]
            ax.imshow(pred, alpha=0.5, cmap='magma')
            if targets is not None:
                tgt = self.sigmoid(targets[idx]).cpu().numpy()[0]
                ax.imshow(tgt, alpha=0.5, cmap='viridis')
            ax.set_title(f'Input @ epoch: {self.epoch} - idx: {idx}')

        self.add_figure(title, fig)
        self.flush()
