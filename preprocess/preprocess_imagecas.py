"""
Preprocess the data for training.
    
"""

import os
import json
import numpy as np
import random
import torch
import nibabel as nib
from scipy.ndimage import zoom
from datasets import ImageCASDataset
from torch.utils.data import DataLoader
import pandas as pd
from drr.drr import create_drr
import imageio
from deepdrr import geo, Volume, MobileCArm
from deepdrr.projector import Projector # separate import for CUDA init
from scipy.spatial.transform import Rotation as R

dataset_info = {
    "mean": -185.3538055419922,
    "std": 439.9675598144531,
}

preprocessing_config = {
        "data_raw_path":        '/home/guests/jorge_padilla/data/ImageCAS',
        "output_folder":        '/home/guests/jorge_padilla/data/ImageCAS/preprocessed',
        "split":                1,
        "normalize_intensity":  True,
        "resample":{
                "active":       True,
                "shape":        [512, 512, 512],
                #"shape":       [256, 256, 256],
                #"shape":        [128, 128, 128],
                "voxel_dim":    [0.25, 0.25, 0.25],
                #"voxel_dim":   [0.5, 0.5, 0.5],
                #"voxel_dim":    [1, 1, 1],
                },
        "generate_drr_projections": True,

        }

        
def applyDRR_nifti(nii_path):
    """
    Create DRRs from a NIfTI image.
    
    Parameters:
    nii_path (string): The path to the image to create the DRRs from.
    axial_rotation (float): The axial rotation angle in degrees.
    coronal_rotation (float): The coronal rotation angle in degrees.
    sagittal_rotation (float): The sagittal rotation angle in degrees.
    
    Returns:
    np.array: The DRRs.
    """
    


    #-------- DiffDRR --------
    # Get the data from the NIfTI image
    #img_data = img_nii.get_fdata()
    # Get the voxel dimensions
    #voxel_dim = img_nii.header.get_zooms()
    #Create DRR
    #drr_axial = create_drr(img_data, spacing=voxel_dim, projection="axial")
    #drr_coronal = create_drr(img_data, spacing=voxel_dim, projection="coronal")
    #drr_sagittal = create_drr(img_data, spacing=voxel_dim, projection="sagittal")

    #-------- DeepDRR --------
    carm = MobileCArm()
    ct = Volume.from_nifti(nii_path)

    with Projector(ct, carm=carm) as projector:
        # Orient and position the patient model in world space.
        axial_rotation = R.from_euler('z', 90, degrees=True) # or R.from_euler('x', 90, degrees=True)
        coronal_rotation = R.from_euler('y', 90, degrees=True) # or R.from_euler('y', 90, degrees=True)
        sagittal_rotation = R.from_euler('x', 90, degrees=True) # or R.from_euler('z', 90, degrees=True)

        # Get axial DRR
        ct.rotate(axial_rotation)
        drr_axial = projector()

        # Undo axial rotation
        ct.rotate(axial_rotation.inv())

        # Get coronal DRR
        ct.rotate(coronal_rotation)
        drr_coronal = projector()

        # Undo coronal rotation
        ct.rotate(coronal_rotation.inv())

        # Get sagittal DRR
        ct.rotate(sagittal_rotation)
        drr_sagittal = projector()

        # Undo sagittal rotation
        ct.rotate(sagittal_rotation.inv())

    return drr_axial, drr_coronal, drr_sagittal


def applyDRRlabel_nifti(label_nii):
    """Creates map depth labels for the DRRs from a NIfTI label image.

    Parameters:
    label_nii (nib.Nifti1Image): The label to create the DRRs from.

    Returns:
    np.array: The DRRs.
    """

    label = label_nii.get_fdata()

    # Set the rotation based on the projection
    axial_rotated_volume = np.rot90(label, k=1, axes=(0, 1))
    coronal_rotated_volume = np.rot90(label, k=1, axes=(0, 2))
    sagittal_rotated_volume = np.rot90(label, k=1, axes=(1, 2))

    # Initialize the depth map
    axial_depth_map = torch.zeros([axial_rotated_volume.shape[0], axial_rotated_volume.shape[1]], dtype=torch.float32)
    coronal_depth_map = torch.zeros([coronal_rotated_volume.shape[0], coronal_rotated_volume.shape[1]], dtype=torch.float32)
    sagittal_depth_map = torch.zeros([sagittal_rotated_volume.shape[0], sagittal_rotated_volume.shape[1]], dtype=torch.float32)

    # Iterate through the slices of the volume and update the depth map
    for i, slice_ in enumerate(axial_rotated_volume):
        # Normalize the depth value to range [0, 1]
        depth_value = i / (axial_rotated_volume.shape[0] - 1)
        # Update the depth map: set the depth value where there is a label (value > 0)
        axial_depth_map[slice_ > 0] = depth_value

    for i, slice_ in enumerate(coronal_rotated_volume):
        # Normalize the depth value to range [0, 1]
        depth_value = i / (coronal_rotated_volume.shape[0] - 1)
        # Update the depth map: set the depth value where there is a label (value > 0)
        coronal_depth_map[slice_ > 0] = depth_value
    
    for i, slice_ in enumerate(sagittal_rotated_volume):
        # Normalize the depth value to range [0, 1]
        depth_value = 1 - i / (sagittal_rotated_volume.shape[0] - 1)
        # Update the depth map: set the depth value where there is a label (value > 0)
        sagittal_depth_map[slice_ > 0] = depth_value
    
    return axial_depth_map, coronal_depth_map, sagittal_depth_map


def remove_outliers_nifti(img):
    """
    Remove outliers from the image.
    
    Parameters:
    img (nib.Nifti1Image): The image to remove outliers from.
    
    Returns:
    nib.Nifti1Image: The image without outliers.
    """

    # Get the data from the NIfTI image
    img_data = img.get_fdata()
    
    # Get the indices of the outliers
    img_outliers = np.where(img_data > dataset_info["mean"] + 3*dataset_info["std"])
    
    # Remove the outliers
    img_data[img_outliers] = dataset_info["mean"] + 3*dataset_info["std"]
    
    # Create a new NIfTI image with the normalized data
    new_header = img.header.copy()
    new_header.set_data_dtype(np.float32)
    img = nib.Nifti1Image(img_data, img.affine, new_header)

    return img


def remove_outliers_drr(drr_axial, drr_coronal, drr_sagittal):
    """
    Remove outliers from the DRRs.
    
    Parameters:
    drr_axial (np.array): The axial DRR.
    drr_coronal (np.array): The coronal DRR.
    drr_sagittal (np.array): The sagittal DRR.
    """

    # Calcculate the mean and standard deviation of the DRRs
    drr_mean = np.mean([drr_axial, drr_coronal, drr_sagittal])
    drr_std = np.std([drr_axial, drr_coronal, drr_sagittal])

    # Get the indices of the outliers
    drr_axial_outliers = np.where(drr_axial > drr_mean + 3*drr_std)
    drr_coronal_outliers = np.where(drr_coronal > drr_mean + 3*drr_std)
    drr_sagittal_outliers = np.where(drr_sagittal > drr_mean + 3*drr_std)

    # Remove the outliers
    drr_axial[drr_axial_outliers] = drr_mean + 3*drr_std
    drr_coronal[drr_coronal_outliers] = drr_mean + 3*drr_std
    drr_sagittal[drr_sagittal_outliers] = drr_mean + 3*drr_std

    return drr_axial, drr_coronal, drr_sagittal


def normalize_intensity_nifti(img):
    """
    Normalize the intensity of the image considering the mean and standard deviation of the dataset.
    
    Parameters:
    img (nib.Nifti1Image): The image to normalize.
    
    Returns:
    nib.Nifti1Image: The normalized image.
    """
        
    # Get the data from the NIfTI image
    img_data = img.get_fdata()
    
    # Normalize the image
    img_data = (img_data - dataset_info["mean"]) / dataset_info["std"]
    
    # Create a new NIfTI image with the normalized data
    new_header = img.header.copy()
    new_header.set_data_dtype(np.float32)
    normalized_nii = nib.Nifti1Image(img_data, img.affine, new_header)
    
    return normalized_nii


def normalize_intensity_drr(drr_axial, drr_coronal, drr_sagittal):
    """
    Normalize the DRRs considering the mean and standard deviation of the drr.
    
    Parameters:
    drr_axial (np.array): The axial DRR.
    drr_coronal (np.array): The coronal DRR.
    drr_sagittal (np.array): The sagittal DRR.
    
    Returns:
    np.array: The normalized DRRs.
    """
    
    # Calculate the mean and standard deviation of the DRRs
    drr_mean = np.mean([drr_axial, drr_coronal, drr_sagittal])
    drr_std = np.std([drr_axial, drr_coronal, drr_sagittal])

    # Normalize the DRRs
    drr_axial = (drr_axial - drr_mean) / drr_std
    drr_coronal = (drr_coronal - drr_mean) / drr_std
    drr_sagittal = (drr_sagittal - drr_mean) / drr_std

    return drr_axial, drr_coronal, drr_sagittal



def resample_nifti(img_nii, label_nii, target_shape, target_voxel_dim):
    """
    Resample the volume to the target shape and voxel dimensions.
    
    Parameters:
    target_shape (tuple): The desired number of voxels (height, width, depth).
    original_voxel_dim (tuple): The original voxel dimensions (dx, dy, dz).
    target_voxel_dim (tuple): The desired voxel dimensions (dx, dy, dz).
    
    Returns:
    nib.Nifti1Image: The resampled NIfTI image.
    """

    # Get the data from the NIfTI images
    img = img_nii.get_fdata()
    label = label_nii.get_fdata()

    # Get the original voxel dimensions
    original_voxel_dim = img_nii.header.get_zooms()

    # Calculate the resampling factor for each dimension
    resample_factor = [o / n for o, n in zip(original_voxel_dim, target_voxel_dim)]

    # Apply the resampling
    img_resampled = zoom(img, resample_factor, order=2)  # order=2 for linear interpolation
    label_resampled = zoom(label, resample_factor, order=0)  # order=0 for nearest-neighbor interpolation

    # If the actual data shape does not match the target shape after resampling,
    # you may need to pad or crop. Here's a simple example of center-cropping/padding:
    cropped_resampled_data = np.zeros(target_shape)
    for i in range(3):  # Iterate over the three dimensions
        if target_shape[i] < img_resampled.shape[i]:  # Crop if needed
            crop_start = (img_resampled.shape[i] - target_shape[i]) // 2
            if i == 0:
                img_resampled = img_resampled[crop_start:crop_start+target_shape[i], :, :]
                label_resampled = label_resampled[crop_start:crop_start+target_shape[i], :, :]
            elif i == 1:
                img_resampled = img_resampled[:, crop_start:crop_start+target_shape[i], :]
                label_resampled = label_resampled[:, crop_start:crop_start+target_shape[i], :]
            else:
                img_resampled = img_resampled[:, :, crop_start:crop_start+target_shape[i]]
                label_resampled = label_resampled[:, :, crop_start:crop_start+target_shape[i]]
        elif target_shape[i] > img_resampled.shape[i]:  # Pad if needed
            pad_amount = (target_shape[i] - img_resampled.shape[i]) // 2
            npad = [(0, 0)] * 3
            npad[i] = (pad_amount, pad_amount)
            img_resampled = np.pad(img_resampled, pad_width=npad, mode='constant', constant_values=0)
            label_resampled = np.pad(label_resampled, pad_width=npad, mode='constant', constant_values=0)

    # Create a new NIfTI image with the resampled data
    new_header = img_nii.header.copy()
    new_header.set_zooms(target_voxel_dim)
    resampled_nii_img = nib.Nifti1Image(img_resampled, img_nii.affine, new_header)

    # Create a new NIfTI image with the resampled data
    new_header = label_nii.header.copy()
    new_header.set_zooms(target_voxel_dim)
    resampled_nii_label = nib.Nifti1Image(label_resampled, label_nii.affine, new_header)

    return resampled_nii_img, resampled_nii_label


def main():
    # -------- Load data --------
    directory = preprocessing_config["data_raw_path"]

    #Translate the split number to the corresponding column index
    split_dict = {1: 'Split-1', 2: 'Split-2', 3: 'Split-3', 4: 'Split-4'}
    split = split_dict[preprocessing_config["split"]]
    
    # Load the data split information
    data_info = pd.read_excel(os.path.join(directory, 'imageCAS_data_split.xlsx'))
    
    # Adjust these column indices if the structure is different
    image_filenames = data_info.iloc[:, 0]
    split_info = data_info[split]
    
    train_indices = split_info[split_info == 'Training'].index
    test_indices = split_info[split_info == 'Testing'].index

    # -------- Set output folder --------
    output_folder = preprocessing_config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTs"), exist_ok=True)

    print(f"Output folder: {output_folder}")

    # -------- Preprocessing train loop --------
    for i, data in enumerate(train_indices):
        print(f"Processing train sample {i+1}/{len(train_indices)}")

        #debugging
        if i == 1:
            break

        #-------- Load data --------
        # Get the image and label file paths
        img_path = os.path.join(directory, f'{image_filenames.iloc[data]}.img.nii.gz')
        label_path = os.path.join(directory, f'{image_filenames.iloc[data]}.label.nii.gz')

        # Load the image and label using nibabel
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)

        #-------- Preprocessing --------
        # Resample the image and label to the desired voxel dimensions
        if preprocessing_config["resample"]["active"]:
            img_nii, label_nii = resample_nifti(img_nii, label_nii, 
                                                preprocessing_config["resample"]["shape"],
                                                preprocessing_config["resample"]["voxel_dim"])
        # Create DRRs
        if preprocessing_config["generate_drr_projections"]:
            drr_axial, drr_coronal, drr_sagittal = applyDRR_nifti(img_path)
            drr_axial, drr_coronal, drr_sagittal = normalize_intensity_drr(drr_axial, drr_coronal, drr_sagittal)
            drr_axial, drr_coronal, drr_sagittal = remove_outliers_drr(drr_axial, drr_coronal, drr_sagittal)
            drr_axial_label, drr_coronal_label, drr_sagittal_label = applyDRRlabel_nifti(label_nii)

        # Normalize the intensity of the image and remove outliers
        if preprocessing_config["normalize_intensity"]:
            img_nii = normalize_intensity_nifti(img_nii)
            img_nii = remove_outliers_nifti(img_nii)

        #-------- Save preprocessed data --------
        # Save as nifti
        nib.save(img_nii, os.path.join(output_folder, "imagesTr", f'{image_filenames.iloc[data]}.img.nii.gz'))
        nib.save(label_nii, os.path.join(output_folder, "labelsTr", f'{image_filenames.iloc[data]}.label.nii.gz'))
        if preprocessing_config["generate_drr_projections"]:
            imageio.imwrite(os.path.join(output_folder, "imagesTr", f'{image_filenames.iloc[data]}_axial.tiff'), drr_axial, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTr", f'{image_filenames.iloc[data]}_coronal.tiff'), drr_coronal, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTr", f'{image_filenames.iloc[data]}_sagittal.tiff'), drr_sagittal, format="tiff")

            imageio.imwrite(os.path.join(output_folder, "labelsTr", f'{image_filenames.iloc[data]}_axial.tiff'), drr_axial_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTr", f'{image_filenames.iloc[data]}_coronal.tiff'), drr_coronal_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTr", f'{image_filenames.iloc[data]}_sagittal.tiff'), drr_sagittal_label, format="tiff")

    # -------- Save preprocessing config --------
    with open(os.path.join(preprocessing_config["output_folder"], "dataset.json"), 'w') as file:
        json.dump(preprocessing_config, file, indent=4)

if __name__ == "__main__":
    main()