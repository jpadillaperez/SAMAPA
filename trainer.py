import os
import torch
import wandb
import random
import datetime
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from datetime import datetime
from utils.config import train_config
from torch.cuda.amp.autocast_mode import autocast

from utils.wandb import save3DImagetoWandb, save2DImagetoWandb

class Trainer(object):

    def __init__(self, model, loss_function, metric, optimizer, scheduler, train_loader, val_loader, config):
        self.model      = model
        self.loss_function = loss_function
        self.metric     = metric
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


    def train(self):
        self.model.to(self.device)
        self.model.train()
        loss_list   = []

        for epoch in range(self.epochs):
            with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
                for i, data in enumerate(self.train_loader):
                    image, label, drr_axial, drr_coronal, drr_sagittal, drr_axial_label, drr_coronal_label, drr_sagittal_label = data

                    # Prepare the input dictionary
                    input = {
                        'image': image.to(self.device),
                        #'label': label.to(self.device),
                        'drr_axial': drr_axial.to(self.device),
                        'drr_coronal': drr_coronal.to(self.device),
                        'drr_sagittal': drr_sagittal.to(self.device),
                        'drr_axial_label': drr_axial_label.to(self.device),
                        'drr_coronal_label': drr_coronal_label.to(self.device),
                        'drr_sagittal_label': drr_sagittal_label.to(self.device)
                    }

                    with autocast():
                        output = self.model(input)
                        #output.requires_grad = True
                        
                    loss = self.loss_function(output, (input['drr_' + train_config["partial_train"][0] + '_label'] > 0).float())

                    # Compute the loss
                    loss.backward()
                    loss_list.append(loss.item())

                    # Backward pass
                    self.optimizer.step()

                    # Zero the gradients
                    self.optimizer.zero_grad()  
                    
                    # Update the progress bar
                    pbar.update()
                    pbar.set_postfix(**{'loss (epoch: {}, batch: {})'.format(epoch, i): loss.item()})

            # Update the learning rate
            self.scheduler.step()

            #log to wandb
            wandb.log({f"{train_config['loss']}Loss": np.mean(loss_list)})
            wandb.log({"LearningRate": self.scheduler.get_last_lr()[0]})
            wandb.log({"Epoch": epoch})

            print(f"{datetime.now()} Training--epoch: {epoch+1}\t"f" lr: {self.scheduler.get_last_lr()[0]:.6f}\t"f" batch loss: {np.mean(loss_list):.6f}\t")

            # Validation

            if (epoch % self.config["val_interval"] == 0):
                # Evaluate the model
                dice_mean = self.evaluation(epoch)

                # Save the model if it is the best so far
                if epoch >= 10 and self.best_dice < dice_mean:
                    self.best_dice = dice_mean
                    self.best_epoch = epoch

                    checkpoint = {
                        'epoch': self.best_epoch, 
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(train_config["output_folder"], "results", "model_" + self.config["model_name"]+ self.config["dataset"] + self.config["partial_train"][0] + '_epoch' + str(epoch) + '_date_' + str(datetime.now()) + '.pth'))
                    print(f"Best epoch: {self.best_epoch}, best dice: {self.best_dice:.4f}")
                    print('Success saving model')

            # Save the model every 20 epochs
            if epoch % 20 == 0:
                checkpoint = {
                    'epoch': epoch, 
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }
                torch.save(checkpoint, os.path.join(self.output_folder, "results", "model_" + self.config["model_name"]+ self.config["dataset"] + self.config["partial_train"][0] + '_epoch' + str(epoch) + '_date_' + str(datetime.now()) + '.pth'))


    def evaluation(self, epoch):
        self.model.eval()
        dice_mean_list  = []
        dice_background_list = []
        dice_vessel_list = []

        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
                for i, data in enumerate(self.val_loader):
                    image, label, drr_axial, drr_coronal, drr_sagittal, drr_axial_label, drr_coronal_label, drr_sagittal_label = data

                    input = {
                        'image': image.to(self.device),
                        'label': label.to(self.device),
                        'drr_axial': drr_axial.to(self.device),
                        'drr_coronal': drr_coronal.to(self.device),
                        'drr_sagittal': drr_sagittal.to(self.device),
                        'drr_axial_label': drr_axial_label.to(self.device),
                        'drr_coronal_label': drr_coronal_label.to(self.device),
                        'drr_sagittal_label': drr_sagittal_label.to(self.device)
                    }

                    with autocast():
                        outputs = self.model(input)
                        outputs.requires_grad = True
                    
                    targets = input['drr_' + train_config["partial_train"][0] + '_label']

                    dice_vessel_score = 1 -self.metric(outputs[:, 0], targets[:, 0])
                    dice_background_score = 1 - self.metric(1 + outputs[:, 0], 1 + targets[:, 0]) # type: ignore
                    dice_mean_score = dice_vessel_score * 0.75 + dice_background_score * 0.25

                    dice_mean_list.append(dice_mean_score.float().cpu().detach().numpy())
                    dice_background_list.append(dice_background_score.float().cpu().detach().numpy())
                    dice_vessel_list.append(dice_vessel_score.float().cpu().detach().numpy())

                    # Save the images to wandb
                    if i == 0:
                        save2DImagetoWandb(outputs[0, 0], f"Val_Output_{train_config['partial_train'][0]}")
                        save2DImagetoWandb(targets[0, 0], f"Val_Target_{train_config['partial_train'][0]}")
        
        print("-"*20)
        print("EVALUATION")
        print(f"dice_average_score: {np.mean(dice_mean_list):0.4f}")
        print(f"dice_background_score: {np.mean(dice_background_list):0.6f}\t"f"dice_vessel_score: {np.mean(dice_vessel_list):0.6f}\t")
        print("-"*20)

        # Log to wandb
        wandb.log({"dice_average_score": np.mean(dice_mean_list)})
        wandb.log({"dice_background_score": np.mean(dice_background_list)})
        wandb.log({"dice_vessel_score": np.mean(dice_vessel_list)})

        return np.mean(dice_mean_list)
    
    def save(self, dice_mean, epoch):
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
    
