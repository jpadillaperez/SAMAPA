import os
import json
import torch
import imageio
import nibabel as nib
import pandas as pd

from utils_preprocess import resample_nifti, applyDRR_nifti, applyDRRlabel_nifti

# Print GPU Info
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")

# -------- Preprocessing config --------
preprocessing_config = {
        "data_raw_path":        '/home/guests/jorge_padilla/data/Task08_HepaticVessel',
        "output_folder":        '/home/guests/jorge_padilla/data/Task08_HepaticVessel/preprocessed',
        "resample":{
                "active":       True,
                #"shape":        [512, 512, 512],
                #"shape":       [256, 256, 256],
                "shape":        [128, 128, 128],
                #"voxel_dim":    [0.25, 0.25, 0.25],
                #"voxel_dim":   [0.5, 0.5, 0.5],
                "voxel_dim":    [1, 1, 1],
                },
        "generate_drr_projections": True,
        }



def main():
    # -------- Load data --------
    directory = preprocessing_config["data_raw_path"]

    # -------- Separate the data into training and testing --------
    train_indices = [f for f in os.listdir(os.path.join(directory, "imagesTr")) if (f.endswith('.nii.gz') and not f.startswith('.'))]
    test_indices = [f for f in os.listdir(os.path.join(directory, "imagesTs")) if (f.endswith('.nii.gz') and not f.startswith('.'))]

    # -------- Set output folder --------
    output_folder = preprocessing_config["output_folder"]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTs"), exist_ok=True)

    print(f"Output folder: {output_folder}")

    # -------- Preprocessing train loop --------
    for i, data in enumerate(train_indices):
        print(f"Processing training sample {data} ({i+1}/{len(train_indices)})")

        #-------- Load the data --------
        img_path = os.path.join(preprocessing_config["data_raw_path"], "imagesTr", data)
        label_path = os.path.join(preprocessing_config["data_raw_path"], "labelsTr", data)
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        id = data.split('.')[0]

        # -------- Resample the data --------
        if preprocessing_config["resample"]["active"]:
            img_nii, label_nii = resample_nifti(img_nii=img_nii, label_nii=label_nii, target_shape=preprocessing_config["resample"]["shape"], target_voxel_dim=preprocessing_config["resample"]["voxel_dim"])

        # -------- Create the DRRs --------
        if preprocessing_config["generate_drr_projections"]:
            drr_axial, drr_coronal, drr_sagittal = applyDRR_nifti(img_path, img_nii.shape[:2])
            drr_axial_label, drr_coronal_label, drr_sagittal_label = applyDRRlabel_nifti(label_nii)

        # -------- Save the data and target --------
        nib.save(img_nii, os.path.join(output_folder, "imagesTr", id + 'img.nii.gz'))
        nib.save(label_nii, os.path.join(output_folder, "labelsTr", id + 'label.nii.gz'))

        if preprocessing_config["generate_drr_projections"]:
            imageio.imwrite(os.path.join(output_folder, "imagesTr", id + '_axial.tiff'), drr_axial, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTr", id + '_coronal.tiff'), drr_coronal, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTr", id + '_sagittal.tiff'), drr_sagittal, format="tiff")

            imageio.imwrite(os.path.join(output_folder, "labelsTr", id + '_axial.tiff'), drr_axial_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTr", id + '_coronal.tiff'), drr_coronal_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTr", id + '_sagittal.tiff'), drr_sagittal_label, format="tiff")


    # -------- Preprocessing test loop --------
    for i, data in enumerate(test_indices):
        print(f"Processing test sample {data} ({i+1}/{len(test_indices)})")

        #-------- Load the data --------
        img_path = os.path.join(preprocessing_config["data_raw_path"], "imagesTs", data)
        img_nii = nib.load(img_path)
        id = data.split('.')[0]

        # -------- Resample the data --------
        if preprocessing_config["resample"]["active"]:
            img_nii, _ = resample_nifti(img_nii=img_nii, label_nii=None, target_shape=preprocessing_config["resample"]["shape"], target_voxel_dim=preprocessing_config["resample"]["voxel_dim"])

        # -------- Create the DRRs --------
        if preprocessing_config["generate_drr_projections"]:
            drr_axial, drr_coronal, drr_sagittal = applyDRR_nifti(img_path, img_nii.shape[:2])

        # -------- Save the data and target --------
        nib.save(img_nii, os.path.join(output_folder, "imagesTs", id + 'img.nii.gz'))

        if preprocessing_config["generate_drr_projections"]:
            imageio.imwrite(os.path.join(output_folder, "imagesTs", id + '_axial.tiff'), drr_axial, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTs", id + '_coronal.tiff'), drr_coronal, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTs", id + '_sagittal.tiff'), drr_sagittal, format="tiff")

if __name__ == "__main__":
    main()