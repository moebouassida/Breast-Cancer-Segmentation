import torch

class Config:
    img_size = 128
    batch_size = 8
    epochs = 10          # small for CI smoke test; increase locally
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow_experiment = "breast_ultrasound_segmentation"
    model_name = "BreastSeg"
    data_path = r'Data/Dataset_BUSI_with_GT/*/*'
