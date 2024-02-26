import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector
from scipy.spatial.transform import Rotation as R
from skimage.transform import resize
from PIL import Image
import torch

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
    for i in reversed(range(label.shape[0])):
        slice_ = axial_rotated_volume[i]
        depth_value = 1 - i / (axial_rotated_volume.shape[0] - 1)
        axial_depth_map[slice_ > 0] = depth_value

    axial_depth_map = axial_depth_map.flip(0).cpu().numpy()

    for i in reversed(range(label.shape[0])):
        slice_ = coronal_rotated_volume[i]
        depth_value = 1 - i / (coronal_rotated_volume.shape[0] - 1)
        coronal_depth_map[slice_ > 0] = depth_value
    
    coronal_depth_map = coronal_depth_map.flip(0).cpu().numpy()
    
    for i in reversed(range(label.shape[0])):
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
    if img_nii is not None:
        img = img_nii.get_fdata()
    if label_nii is not None:
        label = label_nii.get_fdata()

    original_voxel_dim = img_nii.header.get_zooms()
    resample_factor = [o / n for o, n in zip(original_voxel_dim, target_voxel_dim)]

    # Apply the resampling
    img_resampled = zoom(img, resample_factor, order=2)  # order=2 for linear interpolation
    if label_nii is not None:
        label_resampled = zoom(label, resample_factor, order=0)  # order=0 for nearest-neighbor interpolation

    # If the actual data shape does not match the target shape after resampling,
    # you may need to pad or crop. Here's a simple example of center-cropping/padding:
    cropped_resampled_data = np.zeros(target_shape)
    for i in range(3):  # Iterate over the three dimensions
        if target_shape[i] < img_resampled.shape[i]:  # Crop if needed
            crop_amount = (img_resampled.shape[i] - target_shape[i]) / 2
            crop_start = int(np.floor(crop_amount))

            if i == 0:
                img_resampled = img_resampled[crop_start:crop_start+target_shape[i], :, :]
                if label_nii is not None:
                    label_resampled = label_resampled[crop_start:crop_start+target_shape[i], :, :]
            elif i == 1:
                img_resampled = img_resampled[:, crop_start:crop_start+target_shape[i], :]
                if label_nii is not None:
                    label_resampled = label_resampled[:, crop_start:crop_start+target_shape[i], :]
            else:
                img_resampled = img_resampled[:, :, crop_start:crop_start+target_shape[i]]
                if label_nii is not None:
                    label_resampled = label_resampled[:, :, crop_start:crop_start+target_shape[i]]
        
        elif target_shape[i] > img_resampled.shape[i]:  # Pad if needed
            pad_amount = (target_shape[i] - img_resampled.shape[i]) / 2
            npad = [(0, 0)] * 3
            npad[i] = (pad_amount, pad_amount)
            if i == 0:
                img_resampled = np.pad(img_resampled, ((np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int)), (0, 0), (0, 0)), mode='constant', constant_values=0)
                if label_nii is not None:
                    label_resampled = np.pad(label_resampled, ((np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int)), (0, 0), (0, 0)), mode='constant', constant_values=0)
            elif i == 1:
                img_resampled = np.pad(img_resampled, ((0, 0), (np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int)), (0, 0)), mode='constant', constant_values=0)
                if label_nii is not None:
                    label_resampled = np.pad(label_resampled, ((0, 0), (np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int)), (0, 0)), mode='constant', constant_values=0)
            else:
                img_resampled = np.pad(img_resampled, ((0, 0), (0, 0), (np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int))), mode='constant', constant_values=0)
                if label_nii is not None:
                    label_resampled = np.pad(label_resampled, ((0, 0), (0, 0), (np.floor(pad_amount).astype(int), np.ceil(pad_amount).astype(int))), mode='constant', constant_values=0)

    assert tuple(img_resampled.shape) == tuple(target_shape), f"Resampled image shape {img_resampled.shape} does not match target shape {target_shape}"

    # Create a new NIfTI image with the resampled data
    new_header = img_nii.header.copy()
    new_header.set_zooms(target_voxel_dim)
    resampled_nii_img = nib.Nifti1Image(img_resampled, img_nii.affine, new_header)

    # Create a new NIfTI image with the resampled data
    resampled_nii_label = None
    if label_nii is not None:
        new_header = label_nii.header.copy()
        new_header.set_zooms(target_voxel_dim)
        resampled_nii_label = nib.Nifti1Image(label_resampled, label_nii.affine, new_header)

    return resampled_nii_img, resampled_nii_label