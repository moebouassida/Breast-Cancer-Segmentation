import os
import torch


class Config:
    # ── Data ──────────────────────────────────────────────────────
    data_root = os.path.join("Data", "Dataset_BUSI_with_GT")
    mask_suffix = "_mask"
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    train_split = 0.70
    val_split = 0.15
    # test_split = 0.15 (remainder)

    # ── Image ─────────────────────────────────────────────────────
    img_size = 128

    # ── Training ──────────────────────────────────────────────────
    batch_size = 8
    epochs = 100
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Checkpoints ───────────────────────────────────────────────
    checkpoint_dir = "checkpoints"
    save_every = 10
    resume_from = None

    # ── MLflow ────────────────────────────────────────────────────
    mlflow_experiment = "breast_ultrasound_segmentation"
    mlflow_tracking_uri = "file:Experiments/mlruns"
    model_name = "BreastSeg"

    # ── W&B ───────────────────────────────────────────────────────
    wandb_project = "breast-ultrasound-segmentation"
    wandb_entity = None  # set to your W&B username or leave None

    # ── Quality Gates (used by evaluate.py + CI) ──────────────────
    gate_dice = 0.70
    gate_iou = 0.55
    gate_precision = 0.65
    gate_recall = 0.65

    # ── Dataset info (for logging) ────────────────────────────────
    dataset_name = "BUSI"
    dataset_url = "https://scholar.cu.edu.eg/?q=afahmy/pages/dataset"
    notes = ""
