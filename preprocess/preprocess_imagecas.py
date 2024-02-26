import os
import json
import torch
import imageio
import nibabel as nib
import pandas as pd

from utils import resample_nifti, applyDRR_nifti, applyDRRlabel_nifti

# Print GPU Info
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")

# -------- Preprocessing config --------
preprocessing_config = {
        "data_raw_path":        '/home/guests/jorge_padilla/data/ImageCAS',
        "output_folder":        '/home/guests/jorge_padilla/data/ImageCAS/preprocessed_128',
        "split":                1,
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



    # -------- Preprocessing test loop --------

    for i, data in enumerate(test_indices):
        print(f"Processing test sample {i+1}/{len(test_indices)}")

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
        nib.save(img_nii, os.path.join(output_folder, "imagesTs", f'{image_filenames.iloc[data]}.img.nii.gz'))
        nib.save(label_nii, os.path.join(output_folder, "labelsTs", f'{image_filenames.iloc[data]}.label.nii.gz'))

        if preprocessing_config["generate_drr_projections"]:
            imageio.imwrite(os.path.join(output_folder, "imagesTs", f'{image_filenames.iloc[data]}_axial.tiff'), drr_axial, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTs", f'{image_filenames.iloc[data]}_coronal.tiff'), drr_coronal, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "imagesTs", f'{image_filenames.iloc[data]}_sagittal.tiff'), drr_sagittal, format="tiff")

            imageio.imwrite(os.path.join(output_folder, "labelsTs", f'{image_filenames.iloc[data]}_axial.tiff'), drr_axial_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTs", f'{image_filenames.iloc[data]}_coronal.tiff'), drr_coronal_label, format="tiff")
            imageio.imwrite(os.path.join(output_folder, "labelsTs", f'{image_filenames.iloc[data]}_sagittal.tiff'), drr_sagittal_label, format="tiff")


    # -------- Save preprocessing config --------
    with open(os.path.join(preprocessing_config["output_folder"], "dataset.json"), 'w') as file:
        json.dump(preprocessing_config, file, indent=4)

if __name__ == "__main__":
    main()