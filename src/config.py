import torch
import os

class Config:
    # ---- Data ----
    # Root directory containing benign/malignant/normal
    data_root = os.path.join("Data", "Dataset_BUSI_with_GT")
    # Mask naming pattern (BUSI dataset uses '_mask')
    mask_suffix = "_mask"
    # Allowed image extensions
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    
    # ---- Image ----
    img_size = 128  # Resize images/masks to this size
    
    # ---- Training ----
    batch_size = 8
    epochs = 100
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---- Checkpoints ----
    checkpoint_dir = "checkpoints"   
    save_every = 10                   
    resume_from = None               
    
    # ---- MLflow ----
    mlflow_experiment = "breast_ultrasound_segmentation"
    model_name = "BreastSeg"
