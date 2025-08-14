import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os

def log_prediction_visual(image, mask_true, mask_pred, step=0):
    """
    Logs an image with ground truth and predicted masks overlaid to MLflow.

    Args:
        image (np.ndarray): HxWxC image array in [0,1] or [0,255].
        mask_true (np.ndarray): HxW binary ground truth mask.
        mask_pred (np.ndarray): HxW binary predicted mask.
        step (int): Epoch or step number.
    """
    # Normalize image to 0-1 for plotting
    if image.max() > 1:
        image = image / 255.0

    # Make sure masks are binary
    mask_true = (mask_true > 0.5).astype(np.float32)
    mask_pred = (mask_pred > 0.5).astype(np.float32)

    # Create RGB overlays
    gt_overlay = image.copy()
    gt_overlay[..., 1] = np.maximum(gt_overlay[..., 1], mask_true)  # Green for GT

    pred_overlay = image.copy()
    pred_overlay[..., 0] = np.maximum(pred_overlay[..., 0], mask_pred)  # Red for prediction

    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    # Save to a temp file
    os.makedirs("temp_vis", exist_ok=True)
    save_path = f"temp_vis/prediction_step_{step}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    # Log to MLflow
    mlflow.log_artifact(save_path, artifact_path="predictions")
