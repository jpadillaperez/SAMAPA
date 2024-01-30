"""
Preprocess the data for training. Only implemented for the HepaticVessel dataset.
Needed dataset structure:

        data_raw_path
        ├── imagesTr
        │   ├── liver_001.nii.gz
        │   ├── liver_002.nii.gz
        │   ├── liver_003.nii.gz
        │   ├── ...
        │   └── liver_131.nii.gz
        ├── labelsTr
        │   ├── liver_001.nii.gz
        │   ├── liver_002.nii.gz
        │   ├── liver_003.nii.gz
        │   ├── ...
        │   └── liver_131.nii.gz
        ├── imagesTs
        │   ├── liver_001.nii.gz
        │   ├── liver_002.nii.gz
        │   ├── liver_003.nii.gz
        │   ├── ...
        │   └── liver_070.nii.gz
        └── dataset.json

dataset.json example:
        {
            "name": "HepaticVessel",
            "labels": {
                "0": "background",
                "1": "Vessel"
            }
        }
    
"""

import os
import json
import numpy as np
import random
import torch
import nibabel as nib
#from utils.drr.drr import applyDRR
from utils.config import preprocessing_config
from scipy.ndimage import zoom


def save_dataset_preprocessing_info():
    # -------- Create dataset info with preprocessing config --------
    dataset_info = {
        "name": preprocessing_config["name"],
        "labels": preprocessing_config["labels"],
        "normalize_intensity": preprocessing_config["normalize_intensity"],
        "normalize_spacing": preprocessing_config["normalize_spacing"],
        "resize": preprocessing_config["resize"],
        "generate_drr": preprocessing_config["generate_drr"]
    }

    # -------- Save dataset info inside the data_raw_path --------
    with open(os.path.join(preprocessing_config["output_folder"], "dataset.json"), 'w') as file:
        json.dump(dataset_info, file, indent=4)
    


def create_folder_name(dataset_info):
    folder_name = dataset_info["name"]

    if preprocessing_config["normalize_intensity"]:
        folder_name += "_intensity_normalized"

    if preprocessing_config["normalize_spacing"]:
        folder_name += "_spacing_normalized"

    if preprocessing_config["resize"]["active"]:
        folder_name += f"_resized_to_{preprocessing_config['resize']['shape']}"

    if preprocessing_config["generate_drr"]["active"]:
        folder_name += f"_drr_generated"

    folder_name += "_labels_" + "_".join(preprocessing_config["labels"])

    return folder_name


def normalize_spacing(image, spacing):
    # Normalize the spacing to 1mm isotropic
    new_spacing = np.array([1, 1, 1])
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = zoom(image, real_resize_factor, order=1)
    return image, new_spacing


def normalize_intensity(image):
    # Normalize image data to 0-1 range
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

    
def load_dataset_info(json_path):
    with open(json_path, 'r') as file:
        dataset_info = json.load(file)
    return dataset_info


def find_label_key(labels_dict, label_name):
    for key, value in labels_dict.items():
        if value == label_name:
            return key

    raise ValueError(f"Label \"{label_name}\" not found in dataset info.")


def main():    
    # -------- Get Dataset Info --------
    dataset_info = load_dataset_info(os.path.join(preprocessing_config["data_raw_path"], "dataset.json"))

    # -------- Check Dataset --------
    assert dataset_info["name"] in ['HepaticVessel'], 'Other dataset is not implemented'

    #TODO: Add seed to config and use it here

    # ------- Check if cuda is available --------
    if torch.cuda.is_available():
        preprocessing_config["device"] = "cuda"
        print("Using GPU: ", torch.cuda.get_device_name(0))
    else:
        preprocessing_config["device"] = "cpu"
        print("Using CPU")

    # -------- Load data --------
    def get_sorted_files(subdir):
        return sorted([f for f in os.listdir(os.path.join(preprocessing_config["data_raw_path"], subdir)) if not f.startswith('.')])

    train_image_ids = get_sorted_files('imagesTr')
    train_label_ids = get_sorted_files('labelsTr')
    test_image_ids = get_sorted_files('imagesTs')

    # -------- Set output folder --------
    output_folder = os.path.join(preprocessing_config["output_folder"], create_folder_name(dataset_info))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "imagesTs"), exist_ok=True)

    print(f"Output folder: {output_folder}")

    # -------- Preprocessing train loop --------
    for i, data in enumerate(train_image_ids):
        print(f"Processing training sample {data} ({i+1}/{len(train_image_ids)})")

        assert data in train_label_ids, f"Training sample {data} not in train labels"

        # Use Nibabel to load NIfTI files
        img_path = os.path.join(preprocessing_config["data_raw_path"], "imagesTr", data)
        nifti_img = nib.load(img_path)
        img = nifti_img.get_fdata()

        label_path = os.path.join(preprocessing_config["data_raw_path"], "labelsTr", data)
        nifti_label = nib.load(label_path)
        label = nifti_label.get_fdata()

        # Set the target label value
        label_value = find_label_key(dataset_info["labels"], preprocessing_config["labels"])
        label[label != label_value] = 0

        # Normalize image to 0-1 range
        if preprocessing_config["normalize_intensity"]:
            img = normalize_intensity(img)

        # Normalize the spacing
        if preprocessing_config["normalize_spacing"]:
            img, new_spacing = normalize_spacing(img, nifti_img.header.get_zooms())

        # Resize the data and target to the desired input shape using linear interp
        if preprocessing_config["resize"]["active"]:
            original_shape = img.shape
            new_shape = preprocessing_config["resize"]["shape"]

            # Calculate the zoom factors
            zoom_factors = [float(new) / old for new, old in zip(new_shape, original_shape)]

            # Zoom the image
            img_zoomed = zoom(img, zoom_factors, order=1)
            label_zoomed = zoom(label, zoom_factors, order=0)

            # Calculate the padding needed to center the image
            pad_before = [(new - zoomed) // 2 for new, zoomed in zip(new_shape, img_zoomed.shape)]
            pad_after = [new - (zoomed + before) for new, zoomed, before in zip(new_shape, img_zoomed.shape, pad_before)]

            # Apply padding to the zoomed image and label
            img = np.pad(img_zoomed, [(before, after) for before, after in zip(pad_before, pad_after)], mode='constant')
            label = np.pad(label_zoomed, [(before, after) for before, after in zip(pad_before, pad_after)], mode='constant')


        # Save the data and target with the same structure
        #np.save(os.path.join(output_folder, "imagesTr", data[:-7]), img)
        #np.save(os.path.join(output_folder, "labelsTr", data[:-7]), label)

        # Save as nifti
        nifti_img = nib.Nifti1Image(img, nifti_img.affine, nifti_img.header)
        nifti_label = nib.Nifti1Image(label, nifti_label.affine, nifti_label.header)
        nib.save(nifti_img, os.path.join(output_folder, "imagesTr", data))
        nib.save(nifti_label, os.path.join(output_folder, "labelsTr", data))
        
        # If DRR generation is needed
        #if preprocessing_config["generate_drr"]["active"]:
        #    drr_image = applyDRR(volume=img, spacing=nifti_img.header.get_zooms(), 
        #                        axial_rotation=preprocessing_config["generate_drr"]["axial_rotation"], 
        #                        coronal_rotation=preprocessing_config["generate_drr"]["coronal_rotation"], 
        #                        sagittal_rotation=preprocessing_config["generate_drr"]["sagittal_rotation"])


    # -------- Preprocessing test loop --------

    for i, data in enumerate(test_image_ids):
        print(f"Processing test sample {data} ({i+1}/{len(test_image_ids)})")

        # Use Nibabel to load NIfTI files
        img_path = os.path.join(preprocessing_config["data_raw_path"], "imagesTs", data)
        nifti_img = nib.load(img_path)
        img = nifti_img.get_fdata()

        # Normalize image to 0-1 range
        if preprocessing_config["normalize_intensity"]:
            img = normalize_intensity(img)

        # Normalize the spacing
        if preprocessing_config["normalize_spacing"]:
            img, new_spacing = normalize_spacing(img, nifti_img.header.get_zooms())

        # Resize the data and target to the desired input shape using linear interp
        if preprocessing_config["resize"]["active"]:
            zoom_factors = tuple(target_dim / original_dim for target_dim, original_dim in zip(preprocessing_config["resize"]["shape"], img.shape))
            img = zoom(img, zoom_factors, order=1)

        # Save the data and target with the same structure
        np.save(os.path.join(output_folder, "imagesTs", data[:-7]), img)

        # If DRR generation is needed
        #if preprocessing_config["generate_drr"]["active"]:
        #    drr_image = applyDRR(volume=img, spacing=nifti_img.header.get_zooms(), 
        #                        axial_rotation=preprocessing_config["generate_drr"]["axial_rotation"], 
        #                        coronal_rotation=preprocessing_config["generate_drr"]["coronal_rotation"], 
        #                        sagittal_rotation=preprocessing_config["generate_drr"]["sagittal_rotation"])


if __name__ == "__main__":
    main()