"""
Preprocess the data for training.
    
"""

import os
import json
import numpy as np
import random
import torch
import imageio
import nibabel as nib
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import zoom
from skimage.transform import resize
from deepdrr import geo, Volume, MobileCArm
from deepdrr.projector import Projector

# Print GPU Info
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")

# -------- Preprocessing config --------
preprocessing_config = {
        "data_raw_path":        '/home/guests/jorge_padilla/data/ImageCAS',
        "output_folder":        '/home/guests/jorge_padilla/data/ImageCAS/preprocessed',
        "split":                1,
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

def applyDRR_nifti(nii_path, output_shape=(512, 512)):
    """
    Create DRRs from a NIfTI image.
    
    Parameters:
    nii_path (string): The path to the image to create the DRRs from.
    output_shape (tuple): The desired output shape of the DRRs.
    
    Returns:
    np.array: The DRRs.
    """

    #-------- DeepDRR --------
    carm = MobileCArm()
    ct = Volume.from_nifti(nii_path)

    with Projector(ct, carm=carm) as projector:
        # Orient and position the patient model in world space.
        axial_rotation = R.from_euler('z', 90, degrees=True)
        sagittal_rotation = R.from_euler('y', 90, degrees=True)
        coronal_rotation = R.from_euler('x', 90, degrees=True)

        # Center the patient model in world space.
        ct.place_center(carm.isocenter_in_world)

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

    # Resize to original size
    drr_axial = resize(drr_axial, output_shape, order=1)
    drr_coronal = resize(drr_coronal, output_shape, order=1)
    drr_sagittal = resize(drr_sagittal, output_shape, order=1)

    return drr_axial, drr_coronal, drr_sagittal


def applyDRRlabel_nifti(label_nii):
    """Creates map depth labels for the DRRs from a NIfTI label image.

    Parameters:
    label_nii (nib.Nifti1Image): The label to create the DRRs from.

    Returns:
    np.array: The DRRs.
    """

    label = label_nii.get_fdata()
    label = torch.from_numpy(label).cuda()

    # Set the rotation based on the projection
    axial_rotated_volume = torch.rot90(label, k=1, dims=(0, 2))
    coronal_rotated_volume = torch.rot90(label, k=1, dims=(0, 1))
    sagittal_rotated_volume = torch.rot90(label, k=1, dims=(1, 2))

    # Initialize the depth map
    axial_depth_map = torch.zeros([axial_rotated_volume.shape[0], axial_rotated_volume.shape[1]], dtype=torch.float32)
    coronal_depth_map = torch.zeros([coronal_rotated_volume.shape[0], coronal_rotated_volume.shape[1]], dtype=torch.float32)
    sagittal_depth_map = torch.zeros([sagittal_rotated_volume.shape[0], sagittal_rotated_volume.shape[1]], dtype=torch.float32)

    # Iterate through the slices of the volume and update the depth map
    for i in reversed(range(axial_rotated_volume.shape[0])):
        slice_ = axial_rotated_volume[i]
        depth_value = 1 - i / (axial_rotated_volume.shape[0] - 1)
        axial_depth_map[slice_ > 0] = depth_value

    axial_depth_map = axial_depth_map.flip(0).cpu().numpy()

    for i in reversed(range(coronal_rotated_volume.shape[0])):
        slice_ = coronal_rotated_volume[i]
        depth_value = 1 - i / (coronal_rotated_volume.shape[0] - 1)
        coronal_depth_map[slice_ > 0] = depth_value
    
    coronal_depth_map = coronal_depth_map.flip(0).cpu().numpy()
    
    for i in reversed(range(sagittal_rotated_volume.shape[0])):
        slice_ = sagittal_rotated_volume[i]
        depth_value = 1 - i / (sagittal_rotated_volume.shape[0] - 1)
        sagittal_depth_map[slice_ > 0] = depth_value

    sagittal_depth_map = sagittal_depth_map.flip(0).flip(1).cpu().numpy()
    
    return axial_depth_map, coronal_depth_map, sagittal_depth_map


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
    
        #AVOID FIRST 75 SAMPLES
        if i < 75:
            continue
            
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
            drr_axial, drr_coronal, drr_sagittal = applyDRR_nifti(img_path, img_nii.shape[:2])
            drr_axial_label, drr_coronal_label, drr_sagittal_label = applyDRRlabel_nifti(label_nii)

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