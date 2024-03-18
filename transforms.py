import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import nibabel as nib

class NormalizeCustom(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(NormalizeCustom, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        if self.mean is None or self.std is None:
            mean = torch.mean(img)
            std = torch.std(img)
        else:
            mean = self.mean
            std = self.std

        return transforms.Normalize(mean, std)(img)


class RemoveOutliers(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(RemoveOutliers, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        if self.mean is None or self.std is None:
            mean = torch.mean(img)
            std = torch.std(img)
        else:
            mean = self.mean
            std = self.std

        # Remove the outliers
        img[torch.where(img > mean + 3 * std)] = mean + 3 * std

        return img


class InverseIntensity(torch.nn.Module):
    def __init__(self):
        super(InverseIntensity, self).__init__()

    def forward(self, img):
        return torch.max(img) - img