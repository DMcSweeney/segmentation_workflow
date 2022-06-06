"""
Training loop class
"""
import numpy as np
import torch
import torch.nn as nn
import os

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils.Losses import diceLoss
from utils.EarlyStopping import EarlyStopping

class segmenter():
    def __init__(self, model, optimizer, train_loader,
     val_loader, writer, num_epochs, device="cuda:0", output_path='./logs'):
        self.device = torch.device(device)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', verbose=True)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(device)
        self.dsc = diceLoss().to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.scaler = GradScaler()
        self.stopper = EarlyStopping(patience=75)
        #~ Params
        self.num_epochs = num_epochs
        self.best_loss = 10000
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.weights = (1, 1) #* X-ent weight vs DSC weight
        
    def forward(self):
        self.stop = False
        for epoch in range(self.num_epochs+1):
            print(f'Epoch {epoch} of {self.num_epochs}')
            self.writer.epoch = epoch
            self.training(epoch)
            self.validation(epoch)
            self.save_best_model()
            #~ Early stopping
            if self.stop:
                print('==== Early Stopping ====')
                break
    
    def training(self, epoch, writer_step=25):
        self.model.train()
        self.writer.reset_losses()
        for idx, data in enumerate(tqdm(self.train_loader)):
            inputs = data['inputs'].to(self.device, dtype=torch.float32)
            targets = data['targets'].to(self.device, dtype=torch.float32)
            
            #* Zero parameter gradient
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)

                bce_loss = self.bce(outputs['out'], targets)
                dice_loss = self.dsc(outputs['out'], targets)
                train_loss = self.weights[0]*bce_loss + self.weights[1]*dice_loss
            
            #* Compute backward pass on scaled loss
            self.scaler.scale(train_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.writer.metrics['train_loss'].append(train_loss.item())
        
            if epoch % writer_step == 0 and idx == 0:
                print('Plotting inputs...')
                self.writer.plot_inputs('Inputs', inputs)
                self.writer.plot_segmentation(
                    'Predictions', inputs, outputs['out'], targets=targets)

        print('Train Loss:', np.mean(self.writer.metrics['train_loss']))
        self.writer.add_scalar('Training_loss', np.mean(self.writer.metrics['train_loss']), epoch)
        self.writer.add_scalar(
            'Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch, writer_step=5):
        self.model.eval()
        with torch.set_grad_enabled(False):
            print('VALIDATION')
            for batch_idx, data in enumerate(self.val_loader):
                inputs = data['inputs'].to(self.device, dtype=torch.float32)
                targets = data['targets'].to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                bce_loss = self.bce(outputs['out'], targets)
                dice_loss = self.dsc(outputs['out'], targets)
                valid_loss = self.weights[0]*bce_loss + self.weights[1]*dice_loss
                #* Writer
                self.writer.metrics['BCE'].append(self.weights[0]*bce_loss.item())
                self.writer.metrics['DSC'].append(self.weights[1]*dice_loss.item())
                self.writer.metrics['val_loss'].append(valid_loss.item())
                #* --- PLOT TENSORBOARD ---#
                if epoch % writer_step == 0 and batch_idx == 0:
                    self.writer.plot_segmentation('Predictions', inputs, outputs['out'], targets=None)


        print('Validation Loss:', np.mean(self.writer.metrics['val_loss']))
        self.writer.add_scalar('Validation Loss', np.mean(
            self.writer.metrics['val_loss']), epoch)
        self.writer.add_scalar('BCE', np.mean(self.writer.metrics['BCE']), epoch)
        self.writer.add_scalar('DSC', np.mean(self.writer.metrics['DSC']), epoch)
        self.scheduler.step(np.mean(self.writer.metrics['val_loss']))

    def save_best_model(self, model_name='best_model.pt'):
        loss1 = np.mean(self.writer.metrics['val_loss'])
        is_best = loss1 < self.best_loss
        self.best_loss = min(loss1, self.best_loss)
        if is_best:
            print('Saving best model')
            torch.save(self.model.state_dict(),
                       self.output_path + model_name)
        #~ Check stopping criterion
        if self.stopper.step(torch.tensor([loss1])):
            self.stop = True
