train_config = {
        "debug":        False,
        "debug_samples":50,
        "projection":   "axial",                                #options: "axial", "coronal", "sagittal"
        "task":         "depth_map",                            #options: "depth_map", "drr"
        "dataset":      "ImageCAS",                             #options: "ImageCAS", "Task08_HepaticVessel"
        "loss":         "Huber",                                  #options: "MSE", "DiceCE", "Dice", "Huber"
        "metric":       None,                                   #options: None, "Dice", "Precision", "Recall", "SSIM", "PSNR", "RMSE"
        "optimizer":    "Adam",
        "lr":           0.0001,
        "lr_scheduler":  {
                "type":         "StepLR",
                "step_size":    100,
                "gamma":        0.1
        },
        "epochs":       1000,
        "batch_size":   1,
        "num_workers":  1,
        "max_dec_iter": 1,
        "points_per_iter": 1,
        "input_shape":  (128, 128, 128),
        "in_ch":        1,
        "class_num":    1,
        "val_split":    0.2,
        "val_freq":     5,
        "data_path":    "/home/guests/jorge_padilla/data/ImageCAS/preprocessed_128_angio",
        "output_folder":"/home/guests/jorge_padilla/code/Guided_Research/SAMAPA/output",
        "SAM_checkpoint":"/home/guests/jorge_padilla/models/SAM/sam_vit_b_01ec64.pth",
        "verbose":      True
}

sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
                "lr": {"values": [0.1, 0.01, 0.001, 0.0001]},
        }
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