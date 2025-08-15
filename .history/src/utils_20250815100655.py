import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os


def _to_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """Ensure image is float32 in [0,1], shape HxWx3."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        # CxHxW?
        if img.shape[0] in (1, 3) and (img.shape[0] != img.shape[-1]):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {img.shape}, expected 2D or 3D.")

    if img.shape[-1] != 3:
        img_gray = img[..., 0] if img.ndim == 3 else img
        img = np.stack([img_gray, img_gray, img_gray], axis=-1)

    return img


def _to_hw_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is binary float32 HxW."""
    m = mask.astype(np.float32)
    m = np.squeeze(m)
    if m.ndim == 3:
        if m.shape[-1] in (1, 3):
            m = m[..., 0]
        elif m.shape[0] in (1, 3):
            m = m[0, ...]
        else:
            m = m.reshape(m.shape[-2], m.shape[-1])
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape {mask.shape}, expected 2D after squeeze.")
    m = (m > 0.5).astype(np.float32)
    return m


def log_prediction_visual(image, mask_true, mask_pred, step=0):
    """
    Logs an image with ground truth and predicted masks overlaid to MLflow.
    """
    img = _to_rgb_hwc(image)
    gt = _to_hw_mask(mask_true)
    pr = _to_hw_mask(mask_pred)

    H, W = img.shape[:2]
    if gt.shape != (H, W):
        raise ValueError(f"mask_true has shape {gt.shape}, but image is {(H, W)}.")
    if pr.shape != (H, W):
        raise ValueError(f"mask_pred has shape {pr.shape}, but image is {(H, W)}.")

    gt_overlay = img.copy()
    pr_overlay = img.copy()

    gt_overlay[..., 1] = np.maximum(gt_overlay[..., 1], gt)  # G
    pr_overlay[..., 0] = np.maximum(pr_overlay[..., 0], pr)  # R

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(gt_overlay); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pr_overlay); axes[2].set_title("Prediction"); axes[2].axis("off")

    os.makedirs("temp_vis", exist_ok=True)
    save_path = f"temp_vis/prediction_step_{step}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(save_path, artifact_path="predictions")


def log_sample_grid(images, masks_true, masks_pred, max_n=6, step=0):
    """
    Log a grid of N samples (img, GT, Pred) to MLflow.
    images: Tensor/ndarray (N, C, H, W) or (N, H, W, C)
    """
    import torch

    n = min(max_n, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img = images[i]
        gt = masks_true[i]
        pr = masks_pred[i]

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()
            pr = pr.detach().cpu().numpy()

        img = _to_rgb_hwc(img)
        gt = _to_hw_mask(gt)
        pr = _to_hw_mask(pr)

        gt_overlay = img.copy()
        pr_overlay = img.copy()
        gt_overlay[..., 1] = np.maximum(gt_overlay[..., 1], gt)
        pr_overlay[..., 0] = np.maximum(pr_overlay[..., 0], pr)

        axes[i, 0].imshow(img); axes[i, 0].set_title("Original"); axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_overlay); axes[i, 1].set_title("GT"); axes[i, 1].axis("off")
        axes[i, 2].imshow(pr_overlay); axes[i, 2].set_title("Pred"); axes[i, 2].axis("off")

    os.makedirs("temp_vis", exist_ok=True)
    save_path = f"temp_vis/grid_step_{step}.png"
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(save_path, artifact_path="predictions")
