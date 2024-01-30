# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import numpy as np
import nibabel as nib
import random
from utils.config import train_config
from monai.data import CacheDataset, ThreadDataLoader
from einops import rearrange
import torchvision.transforms as transforms
from scipy.ndimage import zoom

class MedicalDataset:
    def __init__(self, config):
        self.config = config
        self.data_dir = config["data_path"]

    def get_loaders(self):
        def get_sorted_files(subdir):
            return sorted([f for f in os.listdir(os.path.join(self.data_dir, subdir)) if not f.startswith('.')])

        train_image_ids = get_sorted_files('imagesTr')
        train_label_ids = get_sorted_files('labelsTr')
        test_image_ids = get_sorted_files('imagesTs')

        train_val_ids = list(zip(train_image_ids, train_label_ids))
        random.shuffle(train_val_ids)

        split_idx = int(len(train_val_ids) * self.config["val_split"])
        train_ids, val_ids = train_val_ids[split_idx:], train_val_ids[:split_idx]

        print('Total Training Samples:', len(train_ids))
        print('Total Validation Samples:', len(val_ids))
        print('Total Test Samples:', len(test_image_ids))

        train_ds = CacheDataset(data=train_ids, transform=None, cache_rate=1.0, num_workers=4)
        val_ds = CacheDataset(data=val_ids, transform=None, cache_rate=1.0, num_workers=4)
        test_ds = CacheDataset(data=test_image_ids, transform=None, cache_rate=1.0, num_workers=4)

        train_loader = ThreadDataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)
        test_loader = ThreadDataLoader(test_ds, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader


    def get_data(self, pack):
        target, sample, spacing = [], []

        if train_config["debugging_mode"]["active"]:
            print("Loading file: ", pack)

        #iterate over the batch size
        for i in range(train_config["batch_size"]):
            # Use Nibabel to load NIfTI files
            data_path = os.path.join(train_config["data_path"], "imagesTr", pack[0])
            nifti_data = nib.load(data_path)
            data = nifti_data.get_fdata()

            target_path = os.path.join(train_config["data_path"], "labelsTr", pack[1])
            nifti_target_data = nib.load(target_path)
            target_data = nifti_target_data.get_fdata()
            
            # Set the target label value
            target_data[target_data != train_config["label_value"]] = 0

            #TODO: Specify the transformations to be applied through the config file
            #TODO: Apply the transformations in preprocessing step

            # Pad the data and target to the desired input shape
            #data = padding(data, train_config["input_shape"])
            #target_data = padding(target_data, train_config["input_shape"])

            # Normalize the spacing
            data, new_spacing = normalize_spacing(data, nifti_data.header.get_zooms())[0]
            target_data, new_spacing  = normalize_spacing(target_data, nifti_target_data.header.get_zooms())[0]
            
            # Append the data and target
            sample.append(data)
            target.append(target_data)
            spacing.append(new_spacing)

        samples = torch.from_numpy(reformat_dim(np.array(sample))).to(torch.float32)
        targets = torch.from_numpy(reformat_dim(np.array(target))).to(torch.float32)
        spacing = torch.from_numpy(np.array(spacing)).to(torch.float32)

        return samples, targets, spacing


    def reformat_dim(self, data):
        """
        Adds a channel dimension to the data if it doesn't have one.
        """
        if data.ndim == 3:  # Assuming [height, width, depth]
            data = data[np.newaxis, ...]  # Add a channel dimension at the start
            data = data[np.newaxis, ...]  # Add a batch dimension at the start
        elif data.ndim == 4:  # Assuming [batch, height, width, depth]
            data = data[np.newaxis, ...]
        elif data.ndim == 5:  # Assuming [batch, channel, height, width, depth]
            pass
        else:
            raise ValueError(f"Unexpected number of dimensions in data: {data.ndim}")
        return data


    def padding(self, sample, input_shape=None):
        """
        Pads the sample to the desired input shape.
        """
        if input_shape is None:
            input_shape = self.config["input_shape"]

        # Check dimensions and adjust padding accordingly
        pad_width_sample = [(0, 0)] * sample.ndim

        # Update padding for spatial dimensions
        for i in range(-3, 0):  # Last 3 dimensions are spatial dimensions
            pad_width_sample[i] = (max((input_shape[i] - sample.shape[i]) // 2, 0), 
                                max((input_shape[i] - sample.shape[i] + 1) // 2, 0))

        sample_padded = np.pad(sample, pad_width_sample, mode='constant', constant_values=0)

        return sample_padded


    def normalize_spacing(self, volume, org_spacing, new_spacing=np.array([1, 1, 1])):
        """
        Normalize the spacing to 1mm x 1mm x 1mm
        """

        from scipy.ndimage import zoom
        resize_factor = org_spacing / new_spacing
        new_real_shape = volume.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / volume.shape
        new_spacing = org_spacing / real_resize_factor
        volume = zoom(volume, real_resize_factor, mode='bilinear')

        print("New shape:", volume.shape)
        print("New spacing:", new_spacing)

        return volume, new_spacing