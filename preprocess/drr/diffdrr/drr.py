# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/00_drr.ipynb.

# %% ../notebooks/api/00_drr.ipynb 3
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from fastcore.basics import patch

from .detector import Detector
from .siddon import siddon_raycast
from .utils import reshape_subsampled_drr

# %% auto 0
__all__ = ['DRR', 'Registration']

# %% ../notebooks/api/00_drr.ipynb 5
class DRR(nn.Module):
    """PyTorch module that computes differentiable digitally reconstructed radiographs."""

    def __init__(
        self,
        volume: np.ndarray,  # CT volume
        spacing: np.ndarray,  # Dimensions of voxels in the CT volume
        sdr: float,  # Source-to-detector radius for the C-arm (half of the source-to-detector distance)
        height: int,  # Height of the rendered DRR
        delx: float,  # X-axis pixel size
        width: int
        | None = None,  # Width of the rendered DRR (if not provided, set to `height`)
        dely: float | None = None,  # Y-axis pixel size (if not provided, set to `delx`)
        p_subsample: float | None = None,  # Proportion of pixels to randomly subsample
        reshape: bool = True,  # Return DRR with shape (b, 1, h, w)
        convention: str = "diffdrr",  # Either `diffdrr` or `deepdrr`, order of basis matrix multiplication
        reverse_x_axis: bool = False,  # If pose includes reflection (in E(3) not SE(3)), reverse x-axis
        patch_size: int
        | None = None,  # Render patches of the DRR in series (useful for large DRRs)
    ):
        super().__init__()

        # Initialize the X-ray detector
        width = height if width is None else width
        dely = delx if dely is None else dely
        self.detector = Detector(
            sdr,
            height,
            width,
            delx,
            dely,
            n_subsample=int(height * width * p_subsample)
            if p_subsample is not None
            else None,
            convention=convention,
            reverse_x_axis=reverse_x_axis,
        )

        # Initialize the volume
        self.register_buffer("spacing", torch.tensor(spacing))
        self.register_buffer("volume", torch.tensor(volume).flip([0]))
        self.reshape = reshape
        self.patch_size = patch_size
        if self.patch_size is not None:
            self.n_patches = (height * width) // (self.patch_size**2)

    def reshape_transform(self, img, batch_size):
        if self.reshape:
            if self.detector.n_subsample is None:
                img = img.view(-1, 1, self.detector.height, self.detector.width)
                #solve mismatch error 
            else:
                img = reshape_subsampled_drr(img, self.detector, batch_size)
        return img

# %% ../notebooks/api/00_drr.ipynb 7
@patch
def forward(self: DRR, rotations: torch.Tensor, translations: torch.Tensor):
    """Generate DRR with rotations and translations parameters."""
    assert len(rotations) == len(translations)
    batch_size = len(rotations)
    source, target = self.detector.make_xrays(
        rotations=rotations,
        translations=translations,
    )

    if self.patch_size is not None:
        n_points = target.shape[1] // self.n_patches
        img = []
        for idx in range(self.n_patches):
            t = target[:, idx * n_points : (idx + 1) * n_points]
            if idx == self.n_patches - 1:
                t = target[:, idx * n_points :]
            partial = siddon_raycast(source, t, self.volume, self.spacing)
            img.append(partial)
        img = torch.cat(img, dim=1)
    else:
        img = siddon_raycast(source, target, self.volume, self.spacing)
    return self.reshape_transform(img, batch_size=batch_size)

# %% ../notebooks/api/00_drr.ipynb 9
class Registration(nn.Module):
    """Perform automatic 2D-to-3D registration using differentiable rendering."""

    def __init__(
        self,
        drr: DRR,
        rotations: torch.Tensor,
        translations: torch.Tensor,
    ):
        super().__init__()
        self.drr = drr
        self.rotations = nn.Parameter(rotations)
        self.translations = nn.Parameter(translations)

    def forward(self):
        return self.drr(self.rotations, self.translations)