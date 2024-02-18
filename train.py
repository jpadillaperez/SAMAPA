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
from monai.losses import DiceCELoss as MonaiDiceCELoss
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def main():    
    # ------- Check if cuda is available --------
    if torch.cuda.is_available():
        train_config["device"] = "cuda"
        print("--------- Using GPU: ", torch.cuda.get_device_name(0))
    else:
        train_config["device"] = "cpu"
        print("--------- Using CPU")

    # -------- Set seeds for reproducibility --------
    torch.autograd.profiler.profile(enabled=True, use_cuda=True)

    SEED = 1
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # -------- Set debugging mode --------
    if train_config["debugging_mode"]["active"]:
        assert train_config["debugging_mode"]["data_path"] is not None, "Debugging mode is on, but no debugging data path is given."
        train_config["data_path"] = train_config["debugging_mode"]["data_path"]
        train_config["epochs"] = train_config["debugging_mode"]["epochs"]
        train_config["val_interval"] = train_config["debugging_mode"]["val_interval"]
        print("------------------ Changing to debugging mode (Ignoring val_num, epochs and using debug_path as data_path)")

    #-------- Initialize Weights and Biases --------
    wandb.init(project="SAMAPAUNet", config=train_config)

    # -------- Load data --------
    if train_config["dataset"] == 'ImageCAS':
        train_dataset = ImageCASDataset(data_path=train_config["data_path"], mode="train")
        train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
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
        #val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
        #test_dataset = HepaticVesselDataset(data_path=train_config["data_path"], mode="test")
        #test_loader = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers=train_config["num_workers"])
        #print("--------- Using HepaticVessel dataset")
    else:
        raise NotImplementedError

    # -------- load model --------
    if train_config["model_name"] == 'SAMAPAUNet':
        print('--------- Using model SAMAPAUNet (with pretrained SAM) and partial training on: ', train_config["partial_train"])
        model = SAMAPAUNet(partial_train=train_config["partial_train"])
    else:
        raise NotImplementedError

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
        loss = MonaiDiceCELoss(to_onehot_y=False)
        print('--------- Using DiceCE Loss')
    else:
        raise NotImplementedError

    # -------- Metrics --------
    if train_config["metric"] == 'Dice':
        metric = DiceLoss()
        print('--------- Using Dice Metric')
    else:
        raise NotImplementedError

    # -------- Optimizers --------
    if train_config["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
        print('--------- Using Adam Optimizer')
    else:
        raise NotImplementedError

    # -------- Learning rate schedulers --------
    if train_config["scheduler"] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 190], gamma=0.1)
        print('--------- Using MultiStepLR Warmup')
    elif train_config["scheduler"] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
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
    print('--------- Starting training ----------\n')
    trainer = Trainer(model, loss, metric, optimizer, lr_scheduler, train_loader, val_loader, train_config)
    trainer.train()

    # -------- Testing ----------
    #trainer.test(test_loader)

    # -------- Finish wandb logging --------
    wandb.finish()

if __name__ == '__main__':
    main()
