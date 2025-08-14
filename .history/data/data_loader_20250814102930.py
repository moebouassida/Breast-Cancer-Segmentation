import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --------- Helpers ---------

def _load_image_gray_resized(path: str, size: int):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def _paired_samples_in_class(class_dir: str, mask_suffix: str, exts):
    """Return list of (image_path, mask_path) pairs for a class dir."""
    if not os.path.isdir(class_dir):
        return []

    img_paths = []
    for ext in exts:
        img_paths += glob(os.path.join(class_dir, f"*{ext}"))

    # keep only images that are NOT masks
    img_paths = [
        p for p in img_paths
        if mask_suffix.lower() not in os.path.basename(p).lower()
    ]

    pairs = []
    for img_path in img_paths:
        base, ext = os.path.splitext(img_path)
        mask_path = f"{base}{mask_suffix}{ext}"
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs

def _infer_root_from_glob(glob_path: str):
    # e.g. "Data/Dataset_BUSI_with_GT/**/*" -> "Data/Dataset_BUSI_with_GT"
    root = glob_path.split("**")[0].rstrip("/\\")
    return root or "."

# --------- Dataset ---------

class BreastUltrasoundSegDataset(Dataset):
    def __init__(self, pairs, size, transform=None):
        self.pairs = pairs
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = _load_image_gray_resized(img_path, self.size)
        mask  = _load_image_gray_resized(mask_path, self.size)
        mask = (mask > 0.5).astype(np.float32)  # ensure binary

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = torch.from_numpy(image).unsqueeze(0)  # (1,H,W)
        mask  = torch.from_numpy(mask).unsqueeze(0)   # (1,H,W)
        return image, mask

# --------- Public API ---------

def get_dataloaders(cfg):
    """
    Expects in cfg:
      - img_size (int)
      - batch_size (int)
      - device ('cuda' or 'cpu')
      - EITHER:
          data_root = 'Data/Dataset_BUSI_with_GT'
        OR
          data_path = 'Data/Dataset_BUSI_with_GT/**/*'  (we'll infer root)
      - OPTIONAL:
          mask_suffix (default: '_mask')
          exts (default: ('.png', '.jpg', '.jpeg', '.bmp'))
    """
    # Resolve root + options
    data_root = getattr(cfg, "data_root", None)
    if data_root is None:
        data_root = _infer_root_from_glob(getattr(cfg, "data_path"))
    data_root = os.path.abspath(data_root)

    mask_suffix = getattr(cfg, "mask_suffix", "_mask")
    exts = getattr(cfg, "exts", (".png", ".jpg", ".jpeg", ".bmp"))

    benign_dir    = os.path.join(data_root, "benign")
    malignant_dir = os.path.join(data_root, "malignant")
    # normal_dir  = os.path.join(data_root, "normal")  # skipped: no masks

    # Build paired samples
    pairs = []
    pairs += _paired_samples_in_class(benign_dir, mask_suffix, exts)
    pairs += _paired_samples_in_class(malignant_dir, mask_suffix, exts)

    print(f"[Data] Root: {data_root}")
    print(f"[Data] Found {len(pairs)} paired image–mask samples (benign + malignant).")

    if len(pairs) == 0:
        raise RuntimeError(
            "No paired samples found. Check folder names (benign/malignant), "
            f"mask suffix ({mask_suffix}), and extensions {exts}."
        )

    # Split into train/val/test
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.2, random_state=42, shuffle=True)
    val_pairs,   test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42, shuffle=True)

    train_ds = BreastUltrasoundSegDataset(train_pairs, cfg.img_size)
    val_ds   = BreastUltrasoundSegDataset(val_pairs,   cfg.img_size)
    test_ds  = BreastUltrasoundSegDataset(test_pairs,  cfg.img_size)

    # Windows: start with num_workers=0; Linux/macOS: try 4
    num_workers = 0 if os.name == "nt" else 4
    pin = (cfg.device == "cuda")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"[Data] Batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    return train_loader, val_loader, test_loader
