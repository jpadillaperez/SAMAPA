import os
import sys
import nibabel as nib
import imageio

sys.path.append("/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/preprocess")

from utils_preprocess import applyDRR_nifti, applyDRRlabel_nifti

img_path = "/home/guests/jorge_padilla/data/ImageCAS/1.img.nii.gz"
label_path = "/home/guests/jorge_padilla/data/ImageCAS/1.label.nii.gz"

img_nii = nib.load(img_path)
label_nii = nib.load(label_path)

drr_axial, drr_coronal, drr_sagittal = applyDRR_nifti(nii_path = img_path, output_shape = img_nii.shape[:2], contrast_path = label_path)

#Save as png
from PIL import Image
import numpy as np

drr_axial = np.array(drr_axial)
drr_coronal = np.array(drr_coronal)
drr_sagittal = np.array(drr_sagittal)

drr_axial = Image.fromarray(drr_axial)
drr_coronal = Image.fromarray(drr_coronal)
drr_sagittal = Image.fromarray(drr_sagittal)

imageio.imwrite("/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/axial.tiff", drr_axial, format="tiff")
imageio.imwrite("/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/coronal.tiff", drr_coronal, format="tiff")
imageio.imwrite("/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/sagittal.tiff", drr_sagittal, format="tiff")
