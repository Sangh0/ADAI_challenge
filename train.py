import os
import logging
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from util.callback import EarlyStopping, CheckPoint
from util.loss import OhemCELoss
from util.metric import Metrics
from util.scheduler import PolynomialLRDecay


class Trainer(object):

    def __init__(
        self, 
        model: nn.Module, 
        num_classes: int,
        lr: float, 
        end_lr: float,
        epochs: int, 
        weight_decay: float,
        miou_loss_weight: float,
        ohem_ce_loss_weight: float,
        lr_scheduling: bool=True,
        check_point: bool=True, 
        early_stop: bool=False,
        train_log_step: int=30, 
        valid_log_step: int=20,
        weight_save_dir='./weights',
    ):
        
        self.logger = logging.getLogger('The logs of training model')
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f'device is {self.device}...')

        self.model = model.to(self.device)
        self.epochs = epochs

        self.loss_func = OhemCELoss(thresh=0.7)
        self.metric = Metrics(n_classes=num_classes, dim=1)
        self.logger.info('loss function ready...')

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        self.logger.info('optimizer ready...')

        self.miou_loss_weight = miou_loss_weight
        self.ohem_ce_loss_weight = ohem_ce_loss_weight
        
        self.lr_scheduling = lr_scheduling
        self.lr_scheduler = PolynomialLRDecay(
            self.optimizer, 
            max_decay_steps=self.epochs,
            end_learning_rate=end_lr,
        )
        self.logger.info('scheduler ready...')

        os.makedirs(weight_save_dir, exist_ok=True)
        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)

        self.early_stop = early_stop
        self.es = EarlyStopping(patience=20, verbose=True, path=weight_save_dir+'/early_stop.pt')
        self.logger.info('callbacks ready...')

        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step
        
        self.writer = SummaryWriter()
        
        self.logger = logging.getLogger('The logs of training model')
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)
            
        self.weight_save_dir = weight_save_dir

    def fit(self, train_loader, valid_loader):  
        self.logger.info('\nStart Training Model...!')
        start_training = time.time()
        for epoch in tqdm(range(self.epochs)):
            init_time = time.time()

            train_loss, train_miou, train_pix_acc = \
                self.train_on_batch(train_loader, epoch)

            valid_loss, valid_miou, valid_pix_acc = \
                self.valid_on_batch(valid_loader, epoch)

            end_time = time.time()

            self.logger.info(f'\n{"="*40} Epoch {epoch+1}/{self.epochs} {"="*40}'
                             f'\n{" "*10}time: {end_time-init_time:.3f}s'
                             f'  lr = {self.optimizer.param_groups[0]["lr"]}')
            self.logger.info(f'train loss: {train_loss:.3f}, train miou: {train_miou:.3f}, train pixel acc: {train_pix_acc:.3f}'
                    f'\nvalid loss: {valid_loss:.3f}, valid miou: {valid_miou:.3f}, valid pixel acc: {valid_pix_acc:.3f}')

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch)
            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = self.weight_save_dir+f'/check_point_{epoch+1}.pt'
                self.cp(1-valid_miou, self.model, path)

            if self.early_stop:
                self.es(valid_loss, self.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        self.writer.close()
        end_training = time.time()
        self.logger.info(f'\nTotal time for training is {end_training-start_training:.2f}s')

        return {
            'model': self.model,
        }

    @torch.no_grad()
    def valid_on_batch(self, valid_loader, epoch):
        self.model.eval()
        batch_loss, batch_miou, batch_pix_acc = 0, 0, 0
        for batch, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs, s2, s3, s4, s5 = self.model(images)
            
            miou = self.metric.mean_iou(outputs, labels)
            pix_acc = self.metric.pixel_acc(outputs, labels)
            batch_miou += miou.item()
            batch_pix_acc += pix_acc.item()
            
            p_loss = self.loss_func(outputs, labels.squeeze())
            a_loss1 = self.loss_func(s2, labels.squeeze())
            a_loss2 = self.loss_func(s3, labels.squeeze())
            a_loss3 = self.loss_func(s4, labels.squeeze())
            a_loss4 = self.loss_func(s5, labels.squeeze())
            total_loss = p_loss + (a_loss1 + a_loss2 + a_loss3 + a_loss4)
            batch_loss += total_loss.item()

            loss = self.miou_loss_weight * (1-miou) + self.ohem_ce_loss_weight * total_loss
            
            if (batch+1) % self.valid_log_step == 0:
                self.logger.info(f'\n{" "*20} Valid Batch {batch+1}/{len(valid_loader)} {" "*20}'
                        f'\nvalid loss: {loss:.3f}, mean IOU: {miou:.3f}, pix accuracy: {pix_acc:.3f}')
            
            step = len(valid_loader) * epoch + batch
            self.writer.add_scalar('Valid/total loss', total_loss.item(), step)
            self.writer.add_scalar('Valid/miou term + total_loss', loss.item(), step)
            self.writer.add_scalar('Valid/principal loss', p_loss.item(), step)
            self.writer.add_scalar('Valid/auxiliary loss 2', a_loss1.item(), step)
            self.writer.add_scalar('Valid/auxiliary loss 3', a_loss2.item(), step)
            self.writer.add_scalar('Valid/auxiliary loss 4', a_loss3.item(), step)
            self.writer.add_scalar('Valid/auxiliary loss 5', a_loss4.item(), step)
            self.writer.add_scalar('Valid/miou', miou.item(), step)
            self.writer.add_scalar('Valid/pixel accuracy', pix_acc.item(), step)

        return batch_loss/(batch+1), batch_miou/(batch+1), batch_pix_acc/(batch+1)


    def train_on_batch(self, train_loader, epoch):
        self.model.train()
        batch_loss, batch_miou, batch_pix_acc = 0, 0, 0
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs, s2, s3, s4, s5 = self.model(images)
            miou = self.metric.mean_iou(outputs, labels)
            pix_acc = self.metric.pixel_acc(outputs, labels)
            batch_miou += miou.item()
            batch_pix_acc += pix_acc.item()
            
            p_loss = self.loss_func(outputs, labels.squeeze())
            a_loss1 = self.loss_func(s2, labels.squeeze())
            a_loss2 = self.loss_func(s3, labels.squeeze())
            a_loss3 = self.loss_func(s4, labels.squeeze())
            a_loss4 = self.loss_func(s5, labels.squeeze())
            # 여기에 total loss 추가함
            total_loss = p_loss + (a_loss1 + a_loss2 + a_loss3 + a_loss4)
            batch_loss += total_loss.item()
            # add miou term
            loss = self.miou_loss_weight * (1-miou) + self.ohem_ce_loss_weight * total_loss
            

            loss.backward()
            self.optimizer.step()

            if (batch+1) % self.train_log_step == 0:
                self.logger.info(f'\n{" "*20} Train Batch {batch+1}/{len(train_loader)} {" "*20}'
                        f'\ntrain loss: {loss:.3f}, mean IOU: {miou:.3f}, pix accuracy: {pix_acc:.3f}')

            step = len(train_loader) * epoch + batch
            self.writer.add_scalar('Train/total loss', total_loss.item(), step)
            # tensorboard log 추가함
            self.writer.add_scalar('Train/miou loss + total loss', loss, step)
            self.writer.add_scalar('Train/principal loss', p_loss.item(), step)
            self.writer.add_scalar('Train/auxiliary loss 2', a_loss1.item(), step)
            self.writer.add_scalar('Train/auxiliary loss 3', a_loss2.item(), step)
            self.writer.add_scalar('Train/auxiliary loss 4', a_loss3.item(), step)
            self.writer.add_scalar('Train/auxiliary loss 5', a_loss4.item(), step)
            self.writer.add_scalar('Train/miou', miou.item(), step)
            self.writer.add_scalar('Train/pixel accuracy', pix_acc.item(), step)

        return batch_loss/(batch+1), batch_miou/(batch+1), batch_pix_acc/(batch+1)
