# General imports
import os
import numpy as np
import wandb
import random
import torch
import nibabel as nib

from datasets import ImageCASDataset, HepaticVesselDataset
from model import SAMAPAUNet
from trainer import Trainer
from utils.config import train_config
from utils.losses import DiceLoss, DiceFocalLoss, DiceLossWeighted, DiceCELoss, BinaryFocalLoss
import torchvision.transforms as transforms

def main():    
    # ------- Check if cuda is available --------
    if torch.cuda.is_available():
        train_config["device"] = "cuda"
        print("Using GPU: ", torch.cuda.get_device_name(0))
    else:
        train_config["device"] = "cpu"
        print("Using CPU")

    # -------- Set seeds for reproducibility --------
    SEED = 1
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # -------- Set debugging mode --------
    if train_config["debugging_mode"]["active"]:
        assert train_config["debugging_mode"]["data_raw_path"] is not None, "Debugging mode is on, but no debugging data path is given."
        train_config["data_raw_path"] = train_config["debugging_mode"]["data_raw_path"]
        train_config["epochs"] = train_config["debugging_mode"]["epochs"]
        train_config["val_interval"] = train_config["debugging_mode"]["val_interval"]
        print("Changed configuration to debugging mode... Ignoring val_num, epochs and using debug_path as data_raw_path")

    #-------- Initialize Weights and Biases --------
    wandb.init(project="SAMAPAUNet", config=train_config)

    # -------- Load data --------
    if train_config["dataset"] == 'ImageCAS':
        train_dataset = ImageCASDataset(data_path=train_config["data_raw_path"], mode="train")
        train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
        train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
        test_dataset = ImageCASDataset(data_path=train_config["data_raw_path"], mode="test")
        test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
        print("Using ImageCAS dataset")
    elif train_config["dataset"] == 'HepaticVessel':
        raise NotImplementedError
        #train_dataset = HepaticVesselDataset(data_path=train_config["data_raw_path"], mode="train")
        #train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
        #train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers=4)
        #val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
        #test_dataset = HepaticVesselDataset(data_path=train_config["data_raw_path"], mode="test")
        #test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
        #print("Using HepaticVessel dataset")
    else:
        raise NotImplementedError

    # -------- load model --------
    if train_config["model_name"] == 'SAMAPAUNet':
        print('Loading model SAMAPAUNet! (with pretrained SAM as Axial Attention Models)')
        model = SAMAPAUNet(train_config['class_num']).to(train_config["device"])
    else:
        raise NotImplementedError

    # -------- Freeze layers --------
    for name, param in model.named_parameters():
        if name.find('MLP_Adapter') != -1:
            param.requires_grad = True
            print('---------Unfreezing layer: ', name)
        elif name.find('Space_Adapter') != -1:
            param.requires_grad = True
            print('---------Unfreezing layer: ', name)
        elif name.find('image_encoder') != -1:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # -------- Loss functions --------
    if train_config["criterion"] == 'Dice':
        criterion = DiceLoss()
        print('---------Using Dice Loss')
    elif train_config["criterion"] == 'DiceCE':
        criterion = DiceCELoss()
        print('---------Using DiceCE Loss')
    else:
        raise NotImplementedError

    # -------- Optimizers --------
    if train_config["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], betas=(0.9, 0.99))
        print('---------Using Adam Optimizer')
    else:
        raise NotImplementedError

    # -------- Learning rate schedulers --------
    if train_config["scheduler"] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 50, 80, 100, 120, 135], gamma=0.1)
        print('---------Using MultiStepLR Warmup')
    elif train_config["scheduler"] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise NotImplementedError

    # -------- Checkpoint resume ----------
    if train_config["resume"] is not None:
        print("loading saved Model...")
        checkpoint  = torch.load(train_config["resume"])
        model.load_state_dict(checkpoint['state_dict'])
        model       = model.to(train_config["device"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch       = checkpoint['epoch']
        print("Model successfully loaded! Current step is: ", epoch)   

    # -------- Training ----------
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, train_loader, val_loader, train_config)
    trainer.run()

    # -------- Testing ----------
    #trainer.test(test_loader)

    # -------- Finish wandb logging --------
    wandb.finish()

if __name__ == '__main__':
    main()
