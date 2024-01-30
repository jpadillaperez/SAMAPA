import os
import pandas as pd
from PIL import Image
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class ImageCASDataset(Dataset):
    def __init__(self, data_path, transform=None, mode="train"):
        """ImageCAS dataset.

        Args:
            data_path (string): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string, optional): Whether to use the training or validation dataset.
        """
        self.data_path = data_path
        self.transform = transform
        self.mode = mode

        if self.mode == "train":
            # Get the image and label file paths for the training dataset filter .img.nii.gz
            nii_path = os.path.join(self.data_path, 'imagesTr')
            files = [f for f in os.listdir(nii_path) if f.endswith('.img.nii.gz')]
            self.nii_files = [os.path.join(nii_path, file) for file in files]
            self.drr_axial_path = [file.replace('.img.nii.gz', '_axial.tiff') for file in self.nii_files]
            self.drr_coronal_path = [file.replace('.img.nii.gz', '_coronal.tiff') for file in self.nii_files]
            self.drr_sagittal_path = [file.replace('.img.nii.gz', '_sagittal.tiff') for file in self.nii_files]

            nii_path = os.path.join(self.data_path, 'labelsTr')
            files = [f for f in os.listdir(nii_path) if f.endswith('.label.nii.gz')]
            self.labels = [os.path.join(nii_path, file) for file in files]

        elif self.mode == "test":
            # Get the image and label file paths for the test dataset
            nii_path = os.path.join(self.data_path, 'imagesTs')
            files = [f for f in os.listdir(nii_path) if f.endswith('.img.nii.gz')]
            self.nii_files = [os.path.join(nii_path, file) for file in files]
            self.drr_axial_path = [file.replace('.img.nii.gz', '_axial.tiff') for file in self.nii_files]
            self.drr_coronal_path = [file.replace('.img.nii.gz', '_coronal.tiff') for file in self.nii_files]
            self.drr_sagittal_path = [file.replace('.img.nii.gz', '_sagittal.tiff') for file in self.nii_files]

            nii_path = os.path.join(self.data_path, 'labelsTs')
            files = [f for f in os.listdir(nii_path) if f.endswith('.label.nii.gz')]
            self.labels = [os.path.join(nii_path, file) for file in files]
        else:
            raise ValueError('Invalid mode! Please use mode="train" or mode="test"')

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.nii_files)

    def __getitem__(self, idx):
        """Return the drr at the given index."""

        nii_path = self.nii_files[idx]
        label_path = self.labels[idx]
        drr_axial_path = self.drr_axial_path[idx]
        drr_coronal_path = self.drr_coronal_path[idx]
        drr_sagittal_path = self.drr_sagittal_path[idx]

        # Load the image and label using nibabel
        image = nib.load(nii_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        drr_axial = Image.open(drr_axial_path)
        drr_coronal = Image.open(drr_coronal_path)
        drr_sagittal = Image.open(drr_sagittal_path)

        # Apply the transforms on the image and label
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        # Convert the numpy arrays to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        drr_axial = torch.from_numpy(drr_axial).float()
        drr_coronal = torch.from_numpy(drr_coronal).float()
        drr_sagittal = torch.from_numpy(drr_sagittal).float()

        return [image, label, drr_axial, drr_coronal, drr_sagittal]


##Instantiate the dataset and dataloader
#print("Creating train and validation datasets...")
#train_dataset = ImageCASDataset(data_path='/home/guests/jorge_padilla/data/ImageCAS/preprocessed', mode="train")
#train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
#val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
#train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#
#print("Creating test dataset...")
#test_dataset = ImageCASDataset(data_path='/home/guests/jorge_padilla/data/ImageCAS/preprocessed', mode="test")
#test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
