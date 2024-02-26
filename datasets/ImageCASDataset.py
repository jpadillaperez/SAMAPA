import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transforms import NormalizeCustom, RemoveOutliers

dataset_info = {
    "stats": {
        "mean": -185.3538055419922,
        "std": 439.9675598144531,
    },
}

class ImageCASDataset(Dataset):
    def __init__(self, data_path, mode="train"):
        """ImageCAS dataset.

        Args:
            data_path (string): Path to the dataset folder.
            mode (string, optional): Whether to use the training or validation dataset.
        """
        self.data_path = data_path
        self.mode = mode

        self.nii_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeCustom(mean=dataset_info["stats"]["mean"], std=dataset_info["stats"]["std"]),
            RemoveOutliers(mean=dataset_info["stats"]["mean"], std=dataset_info["stats"]["std"]),
        ])

        self.drr_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeCustom(),
            RemoveOutliers(),
        ])

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
            self.drr_axial_label_path = [file.replace('.label.nii.gz', '_axial.tiff') for file in self.labels]
            self.drr_coronal_label_path = [file.replace('.label.nii.gz', '_coronal.tiff') for file in self.labels]
            self.drr_sagittal_label_path = [file.replace('.label.nii.gz', '_sagittal.tiff') for file in self.labels]

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
            self.drr_axial_label_path = [file.replace('.label.nii.gz', '_axial.tiff') for file in self.labels]
            self.drr_coronal_label_path = [file.replace('.label.nii.gz', '_coronal.tiff') for file in self.labels]
            self.drr_sagittal_label_path = [file.replace('.label.nii.gz', '_sagittal.tiff') for file in self.labels]

        else:
            raise ValueError('Invalid mode! Please use mode="train" or mode="test"')

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.nii_files)

    def __getitem__(self, idx):
        """Return the image and label for the given index."""
        
        # Load the image and label using nibabel
        image = nib.load(self.nii_files[idx]).get_fdata()
        label = nib.load(self.labels[idx]).get_fdata()

        # Load the DRRs using PIL to numpy
        drr_axial = np.array(Image.open(self.drr_axial_path[idx]))
        drr_coronal = np.array(Image.open(self.drr_coronal_path[idx]))
        drr_sagittal = np.array(Image.open(self.drr_sagittal_path[idx]))
        drr_axial_label = np.array(Image.open(self.drr_axial_label_path[idx]))
        drr_coronal_label = np.array(Image.open(self.drr_coronal_label_path[idx]))
        drr_sagittal_label = np.array(Image.open(self.drr_sagittal_label_path[idx]))

        # Apply the transforms to the images
        image = self.nii_transform(image).unsqueeze(0)
        drr_axial = self.drr_transform(drr_axial).unsqueeze(0)
        drr_coronal = self.drr_transform(drr_coronal).unsqueeze(0)
        drr_sagittal = self.drr_transform(drr_sagittal).unsqueeze(0)

        # Convert the label to a tensor
        label = torch.from_numpy(label).unsqueeze(0)
        drr_axial_label = torch.from_numpy(drr_axial_label).unsqueeze(0)
        drr_coronal_label = torch.from_numpy(drr_coronal_label).unsqueeze(0)
        drr_sagittal_label = torch.from_numpy(drr_sagittal_label).unsqueeze(0)
        
        return [image, label, drr_axial, drr_coronal, drr_sagittal, drr_axial_label, drr_coronal_label, drr_sagittal_label]

def main():
    ##Instantiate the dataset and dataloader
    print("Creating train and validation datasets...")
    train_dataset = ImageCASDataset(data_path='/home/guests/jorge_padilla/data/ImageCAS/preprocessed', mode="train")
    train_dataset, val_dataset = random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print("Creating test dataset...")
    test_dataset = ImageCASDataset(data_path='/home/guests/jorge_padilla/data/ImageCAS/preprocessed', mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    
    print("Train dataset length: ", len(train_dataset))
    print("Val dataset length: ", len(val_dataset))
    print("Test dataset length: ", len(test_dataset))


if __name__ == "__main__":
    main()
    