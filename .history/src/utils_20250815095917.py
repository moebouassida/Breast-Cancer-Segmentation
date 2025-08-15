import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os

def _to_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """Ensure image is float32 in [0,1], shape HxWx3."""
    img = image.astype(np.float32)

    # Normalize if needed
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # Handle dims
    if img.ndim == 2:
        # HxW grayscale -> stack to RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        H, W = None, None
        # channels-first CxHxW?
        if img.shape[0] in (1, 3) and (img.shape[0] != img.shape[-1]):
            # move channels to last
            img = np.transpose(img, (1, 2, 0))  # HWC
        # if single-channel last: HxWx1 -> make RGB
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        # if more than 3 channels, take first 3
        if img.shape[-1] > 3:
            img = img[..., :3]
        # if exactly 3 channels, good
    else:
        raise ValueError(f"Unsupported image shape {img.shape}, expected 2D or 3D.")

    # Final sanity: ensure 3 channels
    if img.shape[-1] != 3:
        # best effort: broadcast grayscale
        img_gray = img[..., 0] if img.ndim == 3 else img
        img = np.stack([img_gray, img_gray, img_gray], axis=-1)

    return img

def _to_hw_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is binary float32 HxW."""
    m = mask.astype(np.float32)
    # squeeze singleton dims
    m = np.squeeze(m)
    if m.ndim == 3:
        # If 3D, pick a channel (prefer last if looks like HxWxC)
        if m.shape[-1] in (1, 3):  # HxWxC
            m = m[..., 0]
        elif m.shape[0] in (1, 3):  # CxHxW
            m = m[0, ...]
        else:
            # Ambiguous; try to pick the middle 2 dims as spatial
            # Here we assume last two are HxW
            m = m.reshape(m.shape[-2], m.shape[-1])
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape {mask.shape}, expected 2D after squeeze.")
    # binarize
    m = (m > 0.5).astype(np.float32)
    return m

def log_prediction_visual(image, mask_true, mask_pred, step=0):
    """
    Logs an image with ground truth and predicted masks overlaid to MLflow.

    Args:
        image (np.ndarray): image array (H,W,C) or (C,H,W) or (H,W) or (H,W,1) in [0,1] or [0,255].
        mask_true (np.ndarray): HxW or broadcastable to HxW (binary or logits).
        mask_pred (np.ndarray): HxW or broadcastable to HxW (binary or logits).
        step (int): Epoch or step number.
    """
    # Normalize image and shapes
    img = _to_rgb_hwc(image)
    gt = _to_hw_mask(mask_true)
    pr = _to_hw_mask(mask_pred)

    # Align masks to image size if needed
    H, W = img.shape[:2]
    if gt.shape != (H, W):
        raise ValueError(f"mask_true has shape {gt.shape}, but image is {(H, W)}.")
    if pr.shape != (H, W):
        raise ValueError(f"mask_pred has shape {pr.shape}, but image is {(H, W)}.")

    # Create overlays (non-destructive)
    gt_overlay = img.copy()
    pr_overlay = img.copy()

    # Green for GT, Red for prediction (via max so it 'lights up' the channel)
    gt_overlay[..., 1] = np.maximum(gt_overlay[..., 1], gt)  # G
    pr_overlay[..., 0] = np.maximum(pr_overlay[..., 0], pr)  # R

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pr_overlay)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")

    # Save to a temp file
    os.makedirs("temp_vis", exist_ok=True)
    save_path = f"temp_vis/prediction_step_{step}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(save_path, artifact_path="predictions")
