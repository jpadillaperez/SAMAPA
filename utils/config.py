train_config = {
        "model_name":   "SAMAPAUNet",
        "dataset":      "ImageCAS",
        "data_path":    "/home/guests/jorge_padilla/data/HepaticVessel/Task08_HepaticVessel", 
        "scheduler":    "MultiStepLR",
        "criterion":    "Dice",
        "optimizer":    "Adam",
        "lr":           0.5,
        "epochs":       30,
        "val_interval": 10,
        "batch_size":   1,
        "input_shape":  (128, 128, 128),
        "in_ch":        1,
        "class_num":    1,
        "val_split":    0.2,
        "resume":       None,
        "label_value":  1,
        "debugging_mode": {
                "active":        True,
                "data_path":    "/home/guests/jorge_padilla/data/HepaticVessel/Task08_HepaticVessel",
                "epochs": 30,
                "val_interval": 10,
                },
        "output_folder":"/home/guests/jorge_padilla/code/Guided_Research/SAMAPAUNet/output",
        "SAM_checkpoint":"/home/guests/jorge_padilla/models/SAM/sam_vit_b_01ec64.pth"
}

test_config = {
        "model_name":   "APAUNet",
        "dataset":     "Task08_HepaticVessel",
        "data_dir":    "/data/jorge_perez/Task08_HepaticVessel_preprocessed/08_3d/Test",
        "in_ch":        1,
        "class_num":    1,
        "batch_size":   2,
        "input_shape":  (80, 80, 80),
        "resume":       "/data/jorge_perez/Models/model_APAUNetHepaticVessel_epoch200_DICE_107817.pth",
        "use_cuda":     True,
}