import numpy as np
import os
import math
import itertools
from .diffdrr.drr import DRR
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")


def create_drr(volume, spacing, projection = "axial"):
    """
    Creates a DRR from a volume.
    """
    #Normalize values
    volume[volume <= -1000] = -1000
    volume = volume + 1000

    #Create DRR
    drr = DRR(
        volume,
        spacing=spacing,
        sdr=100000,
        width=volume.shape[0],
        delx=spacing[0] * 2,
        height=volume.shape[0],
        dely=spacing[0] * 2,
        reverse_x_axis=False,
        patch_size=int(volume.shape[0]/4),
    ).to(device)

    if projection == "axial":
        drr_rotations = torch.tensor([[torch.pi/2, 0.0, 0.0]], device=device)
    elif projection == "coronal":
        drr_rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    elif projection == "sagittal":
        drr_rotations = torch.tensor([[0.0, torch.pi/2, 0.0]], device=device)
    else:
        raise ValueError("The projection is invalid")

    bx, by, bz = torch.tensor(volume.shape) * torch.tensor(spacing) / 2
    drr_translations = torch.tensor([[bx, by, bz]], device=device)

    result = drr(drr_rotations, drr_translations).squeeze(0).squeeze(0).cpu().numpy()

    return result