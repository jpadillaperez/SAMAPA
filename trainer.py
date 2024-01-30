# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import datetime
import torch
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from utils.config import train_config
from dataset import get_data
from datetime import datetime
import SimpleITK as sitk
from tqdm import tqdm
import random

class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler, train_loader, val_loader, config):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config     = config
        self.device     = self.config["device"]
        self.epochs     = self.config["epochs"]
        self.class_num = self.config["class_num"]
        self.best_dice  = 0 
        self.best_epoch = 0

        self.output_folder = self.config["output_folder"]


    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            if (epoch % self.config["val_interval"] == 0):
                dice_mean = self.evaluation(epoch)
                if epoch >= 10 and self.best_dice < dice_mean: #SAVING BEST MODEL
                    self.save(dice_mean, epoch) 
            if epoch % 20 == 0: #SAVING MODEL EVERY 100 EPOCHS
                checkpoint = {
                    'epoch': epoch, 
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }
                torch.save(checkpoint, os.path.join(self.output_folder, "results", "model_" + self.config["model_name"]+ self.config["dataset"] + '_epoch' + str(epoch) + '_date_' + str(datetime.now()) + '.pth'))

    def train(self, epoch):
        self.model.train()
        loss_list   = []

        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
            #iterate over the batches of the dataset using the threadDataLoader
            for i, data in enumerate(train_dataloader):
                # Unpack the data
                images, labels, drr_axial, drr_coronal, drr_sagittal = data
                images, labels = images.to(self.device), labels.to(self.device)
                drr_axial, drr_coronal, drr_sagittal = drr_axial.to(self.device), drr_coronal.to(self.device), drr_sagittal.to(self.device)

                # Prepare the sample dictionary
                sample = {
                    'image': images,
                    'masks': labels,
                    'drr_axial': drr_axial,
                    'drr_coronal': drr_coronal,
                    'drr_sagittal': drr_sagittal,
                    'device': self.device
                }

                with autocast():
                    output = self.model(sample)
                    loss = self.criterion(output, labels) # type: ignore
                    loss.backward()
                    self.optimizer.step()
                
                pbar.update()
                pbar.set_postfix(**{'loss (epoch: {}, batch: {})'.format(epoch, i): loss.item()})

                self.optimizer.zero_grad()                  
                loss_list.append(loss.item())

                del inputs, labels, sample, output
                torch.cuda.empty_cache()
            
        self.scheduler.step()
        print("-"*20)
        print(f"{datetime.now()} Training--epoch: {epoch+1}\t"f" lr: {self.scheduler.get_last_lr()[0]:.6f}\t"f" batch loss: {np.mean(loss_list):.6f}\t")





    def evaluation(self, epoch):
        # a trick for quick evaluate on validation set, the formal evaluate should use slide_windows_inference in MONAI.
        self.model.eval()
        dice_mean_list  = []
        dice_background_list = []
        dice_vessel_list = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
                for pack in self.val_loader:
                    with autocast():
                        imgs, targets, spacings = get_data(pack)
                        imgs, targets, spacings = imgs.to(self.device), targets.to(self.device), spacings.to(self.device)

                        sample = {}
                        sample['image'] = imgs
                        sample['masks'] = targets
                        sample['device'] = self.device
                        sample['spacings'] = spacings
                        
                        outputs = self.model(sample)
                        loss    = self.criterion(outputs, targets)   

                    dice_vessel_score = 1 - self.criterion(outputs[:, 0], targets[:, 0]) # type: ignore
                    dice_background_score = 1 - self.criterion(1 + outputs[:, 0], 1 + targets[:, 0]) # type: ignore
                    dice_mean_score = dice_vessel_score * 0.75 + dice_background_score * 0.25

                    dice_mean_list.append(dice_mean_score.float().cpu().detach().numpy())
                    dice_background_list.append(dice_background_score.float().cpu().detach().numpy())
                    dice_vessel_list.append(dice_vessel_score.float().cpu().detach().numpy())
        
        print("-"*20)
        print("EVALUATION")
        print(f"dice_average_score: {np.mean(dice_mean_list):0.4f}")
        print(f"dice_background_score: {np.mean(dice_background_list):0.6f}\t"f"dice_vessel_score: {np.mean(dice_vessel_list):0.6f}\t")
        print("-"*20)
        return np.mean(dice_mean_list)
    
    def save(self, dice_mean, epoch):
        self.best_dice = dice_mean
        self.best_epoch = epoch
        checkpoint = {
            'epoch': epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
                    }
        torch.save(checkpoint, os.path.join(train_config["output_folder"], "results", "model_" + self.config["model_name"]+ self.config["dataset"] + '_epoch' + str(epoch) + '_date_' + str(datetime.now()) + '.pth')) # type: ignore
        print(f"best epoch: {self.best_epoch}, best dice: {self.best_dice:.4f}")
        print('Success saving model')

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def generate_click_prompt(self, img, msk, pt_label = 1):
        # return: prompt, prompt mask
        pt_list = []
        msk_list = []
        b, c, h, w, d = msk.size()
        msk = msk[:,0,:,:,:]
        for i in range(d):
            pt_list_s = []
            msk_list_s = []
            for j in range(b):
                msk_s = msk[j,:,:,i]
                indices = torch.nonzero(msk_s)
                if indices.size(0) == 0:
                    # generate a random array between [0-h, 0-h]:
                    random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                    new_s = msk_s
                else:
                    random_index = random.choice(indices)
                    label = msk_s[random_index[0], random_index[1]]
                    new_s = torch.zeros_like(msk_s)
                    # convert bool tensor to int
                    new_s = (msk_s == label).to(dtype = torch.float)
                    # new_s[msk_s == label] = 1
                pt_list_s.append(random_index)
                msk_list_s.append(new_s)
            pts = torch.stack(pt_list_s, dim=0)
            msks = torch.stack(msk_list_s, dim=0)
            pt_list.append(pts)
            msk_list.append(msks)
        pt = torch.stack(pt_list, dim=-1)
        msk = torch.stack(msk_list, dim=-1)

        msk = msk.unsqueeze(1)

        return img, pt, msk #[b, 2, d], [b, c, h, w, d]
    
