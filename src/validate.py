import torch
import mlflow
import wandb
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.metrics import compute_all_metrics, dice_score, iou_score


def _threshold_sweep(logits: torch.Tensor, masks: torch.Tensor, device: str):
    thresholds = [i / 20 for i in range(1, 20)]
    probs = torch.sigmoid(logits.to(device))
    masks = masks.to(device)
    dice_vals, iou_vals = [], []
    with torch.no_grad():
        for t in thresholds:
            preds = (probs > t).float()
            dice_vals.append(dice_score(preds, masks).item())
            iou_vals.append(iou_score(preds, masks).item())
    return thresholds, dice_vals, iou_vals


def _make_overlay(img_np: np.ndarray, mask_np: np.ndarray, color="red") -> np.ndarray:
    """Overlay a binary mask on a grayscale image."""
    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    overlay = rgb.copy()
    ch = {"red": 0, "green": 1, "blue": 2}.get(color, 0)
    overlay[..., ch] = np.maximum(overlay[..., ch], mask_np * 0.6)
    return np.clip(overlay, 0, 1)


def validate(
    model: torch.nn.Module,
    val_loader,
    device: str,
    epoch: int = 0,
    log_images: bool = True,
    use_wandb: bool = True,
) -> dict:
    """
    Run validation loop, compute all metrics, log to MLflow + W&B.
    Returns dict of averaged metrics.
    """
    model.eval()
    accumulated = {
        k: [] for k in ["dice", "iou", "pixel_accuracy", "precision", "recall", "f1"]
    }
    first_batch = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            batch_metrics = compute_all_metrics(preds, masks)
            for k, v in batch_metrics.items():
                accumulated[k].append(v)

            if first_batch is None:
                first_batch = (
                    images.detach().cpu(),
                    masks.detach().cpu(),
                    preds.detach().cpu(),
                    logits.detach().cpu(),
                )

    # Average metrics
    avg = {k: round(sum(v) / len(v), 4) for k, v in accumulated.items()}

    # ── Log to MLflow ─────────────────────────────────────────────
    for k, v in avg.items():
        mlflow.log_metric(f"val_{k}", v, step=epoch)

    # ── Log to W&B ────────────────────────────────────────────────
    if use_wandb and wandb.run is not None:
        wandb.log({f"val/{k}": v for k, v in avg.items()}, step=epoch)

    # ── Visual Logging ────────────────────────────────────────────
    if log_images and first_batch is not None:
        imgs, masks_cpu, preds_cpu, logits_cpu = first_batch
        n = min(4, imgs.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n):
            img_np = imgs[i].squeeze().numpy()
            gt_np = masks_cpu[i].squeeze().numpy()
            pr_np = preds_cpu[i].squeeze().numpy()

            axes[i, 0].imshow(img_np, cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(_make_overlay(img_np, gt_np, "green"))
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(_make_overlay(img_np, pr_np, "red"))
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        fig.suptitle(f"Epoch {epoch} — Dice: {avg['dice']:.4f} | IoU: {avg['iou']:.4f}")
        fig.tight_layout()

        os.makedirs("temp_vis", exist_ok=True)
        save_path = f"temp_vis/val_epoch_{epoch}.png"
        fig.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

        mlflow.log_artifact(save_path, artifact_path="predictions")

        if use_wandb and wandb.run is not None:
            wandb.log({"val/predictions": wandb.Image(save_path)}, step=epoch)

        # Threshold sweep
        ths, dvals, ivals = _threshold_sweep(logits_cpu, masks_cpu, "cpu")
        fig2, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ths, dvals, label="Dice", marker="o", markersize=3)
        ax.plot(ths, ivals, label="IoU", marker="s", markersize=3)
        ax.axvline(
            x=0.5, color="gray", linestyle="--", alpha=0.5, label="threshold=0.5"
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold Sweep — Epoch {epoch}")
        ax.legend()
        fig2.tight_layout()
        sweep_path = f"temp_vis/threshold_sweep_{epoch}.png"
        fig2.savefig(sweep_path, bbox_inches="tight")
        plt.close(fig2)
        mlflow.log_artifact(sweep_path, artifact_path="threshold_sweeps")

        if use_wandb and wandb.run is not None:
            wandb.log({"val/threshold_sweep": wandb.Image(sweep_path)}, step=epoch)

    return avg
