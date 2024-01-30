import os
import imageio
from utils.config import train_config
import wandb
import torch
import numpy as np


output_dir = os.path.join(train_config["output_folder"], "media")

def save3DImagetoWandb(volume, filename):
    # Create a GIF from a sequence of images
    images = []  # Your list of images (as numpy arrays)

    #check if volume is a tensor on the GPU
    if volume.is_cuda:
        volume = volume.cpu()
    #check if volume is a tensor
    if torch.is_tensor(volume):
        volume = volume.detach().numpy()
    #check if volume is a numpy array
    if isinstance(volume, np.ndarray):
        #properly convert to uint8
        volume = volume.astype(np.float32)
        volume -= volume.min()
        volume /= volume.max()
        volume = (volume*255).astype(np.uint8)
    #change orientation to match wandb
    #volume = np.moveaxis(volume, 0, -1)

    for img in volume:
        images.append(img)
    
    gif_path = os.path.join(output_dir, f'{filename}.gif')
    
    # Save the GIF
    imageio.mimsave(gif_path, images, fps=5)
    
    # Log the GIF to wandb
    wandb.log({filename: wandb.Image(gif_path)})

def save2DImagetoWandb(image, filename):
    #check if image is a tensor on the GPU
    if image.is_cuda:
        image = image.cpu()
    #check if image is a tensor
    if torch.is_tensor(image):
        image = image.detach().numpy()

    #check if image is a numpy array
    if isinstance(image, np.ndarray):
        #properly convert to uint8
        image = image.astype(np.float32)
        image -= image.min()
        image /= image.max()
        image = (image*255).astype(np.uint8)
    
    # Log the GIF to wandb
    wandb.log({filename: wandb.Image(image)})


