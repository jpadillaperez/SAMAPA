import os
import numpy as np
import wandb
import random
import torch
import yaml
import nibabel as nib
import pytorch_lightning as pl
import torch.nn as nn
from monai.losses import DiceCELoss, DiceCELoss
from torchmetrics import Dice, Precision, Recall, F1Score
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets import ImageCASDataset, HepaticVesselDataset
from utils.config import train_config, sweep_config
from utils.wandb import save2DImagetoWandb
from model.SAMAPA import SAMAPA
from train import SAMAPA_trainer

def main():
    # ------- Set seeds for reproducibility --------
    pl.seed_everything(1)
    torch.set_float32_matmul_precision('medium')

    # ------- Check if cuda is available --------
    if torch.cuda.is_available():
        train_config["device"] = "cuda"
        print("--------- Using GPU: ", torch.cuda.get_device_name(0))
    else:
        train_config["device"] = "cpu"
        print("--------- Using CPU")

    # ------- Initialize Weights and Biases --------
    wandb.init(config=train_config)
    wandb_logger = WandbLogger(project="SAMAPAUNet", config=wandb.config)

    for key, value in wandb.config.items():
        train_config[key] = value

    print("--------- Using sweep config: ", train_config)

    # -------- Load data --------
    if train_config["dataset"] == 'ImageCAS':
        train_dataset = ImageCASDataset(data_path=train_config["data_path"], mode="train")
        train_dataset, val_dataset = random_split(train_dataset, [int((1 - train_config["val_split"]) * len(train_dataset)), int(train_config["val_split"] * len(train_dataset))])

        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=train_config["num_workers"])
        val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
        
        test_dataset = ImageCASDataset(data_path=train_config["data_path"], mode="test")
        test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
        print("--------- Using ImageCAS dataset")
    elif train_config["dataset"] == 'HepaticVessel':
        raise NotImplementedError
        #train_dataset = HepaticVesselDataset(data_path=train_config["data_path"], mode="train")
        #train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
        #train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=train_config["num_workers"])
        #print("--------- Using HepaticVessel dataset")
    else:
        raise NotImplementedError

    # ------- Initialize model --------
    model = SAMAPA(train_config)

    # -------- Load pretrained SAM --------
    if train_config["SAM_checkpoint"] is not None:
        print('------------------ Loading pretrained SAM from: ', train_config["SAM_checkpoint"])
        checkpoint = torch.load(train_config["SAM_checkpoint"])
        for name, param in model.named_parameters():
            if name.find('image_encoder') != -1:
                key = "image_encoder." + name.split('image_encoder.')[1]
                if key.find('MLP_Adapter') != -1 or key.find('Space_Adapter') != -1:
                    continue
                param.data = checkpoint[key]
            elif name.find('prompt_encoder') != -1:
                key = "prompt_encoder." + name.split('prompt_encoder.')[1]
                param.data = checkpoint[key]
            elif name.find('mask_decoder') != -1:
                key = "mask_decoder." + name.split('mask_decoder.')[1]
                param.data = checkpoint[key]

    # -------- Freeze layers --------
    print('------------------ Freezing layers')
    for name, param in model.named_parameters():
        if name.find('MLP_Adapter') != -1:
            param.requires_grad = True
        elif name.find('Space_Adapter') != -1:
            param.requires_grad = True
        elif name.find('image_encoder') != -1:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # -------- Loss functions --------
    if train_config["loss"] == 'Dice':
        loss = DiceLoss()
        print('--------- Using Dice Loss')
    elif train_config["loss"] == 'DiceCE':
        loss = DiceCELoss(to_onehot_y=False)
        print('--------- Using DiceCE Loss')
    else:
        raise NotImplementedError

    # -------- Metrics --------
    metrics = []
    for metrics_comp in train_config["metric"]:
        if 'Dice' in metrics_comp:
            metrics.append(Dice().to(train_config["device"]))
            print('--------- Using Dice metric')
        if 'Precision' in metrics_comp:
            metrics.append(Precision(task="binary").to(train_config["device"]))
            print('--------- Using Precision metric')
        if 'Recall' in metrics_comp:
            metrics.append(Recall(task="binary").to(train_config["device"]))
            print('--------- Using Recall metric')
        if 'F1Score' in metrics_comp:
            metrics.append(F1Score(task="binary").to(train_config["device"]))
            print('--------- Using F1Score metric')
        if metrics_comp not in ['Dice', 'Precision', 'Recall', 'F1Score']:
            raise NotImplementedError

    metrics = {metric.__class__.__name__: metric for metric in metrics}

    # -------- Initialize trainer --------
    model = SAMAPA_trainer(train_config, model, loss, metrics)

    # -------- Checkpoint callback --------
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', save_last=True, dirpath=wandb_logger.experiment.dir + '/output/checkpoints')

    # -------- Learning rate monitor --------
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # -------- Trainer --------
    trainer = pl.Trainer(   accelerator=train_config["device"],
                            devices= 1 if train_config["device"] == "cuda" else 0,
                            max_epochs=train_config["epochs"],
                            callbacks=[checkpoint_callback, lr_monitor],
                            logger=wandb_logger,
                            log_every_n_steps=16,
                            val_check_interval=train_config["val_freq"],

                            )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
    


if __name__ == "__main__":
    # -------- Initialize Weights and Biases --------
    sweep_id = wandb.sweep(sweep=sweep_config, project="SAMAPAUNet")
    wandb.agent(sweep_id, function=main)