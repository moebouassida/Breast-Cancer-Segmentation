import torch
import mlflow
from src.metrics import dice_score, iou_score, pixel_accuracy, precision_score, recall_score
from src.utils import log_prediction_visual, log_sample_grid


def _threshold_sweep_for_batch(logits, masks, device, thresholds=None):
    """
    Do a simple threshold sweep on the first batch to visualize
    how dice/iou vary with threshold.
    """
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]  # 0.05..0.95

    with torch.no_grad():
        probs = torch.sigmoid(logits)
        dice_vals, iou_vals = [], []
        for t in thresholds:
            preds = (probs > t).float()
            dice_vals.append(dice_score(preds, masks).item())
            iou_vals.append(iou_score(preds, masks).item())
    return thresholds, dice_vals, iou_vals


def validate(model, val_loader, device, log_images=False, epoch=None, do_threshold_sweep=True):
    model = model.to(device)
    model.eval()

    dices, val_iou, val_pixel_acc, val_precision, val_recall = [], [], [], [], []
    first_batch_cache = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            outputs = torch.sigmoid(logits)
            preds = (outputs > 0.5).float()

            dices.append(dice_score(preds, masks).item())
            val_iou.append(iou_score(preds, masks).item())
            val_pixel_acc.append(pixel_accuracy(preds, masks).item())
            val_precision.append(precision_score(preds, masks).item())
            val_recall.append(recall_score(preds, masks).item())

            # Cache first batch for visuals and threshold sweep
            if first_batch_cache is None:
                first_batch_cache = (images.detach().cpu(), masks.detach().cpu(), preds.detach().cpu(), logits.detach().cpu())

            # Log only first sample overlay for quick glance
            if log_images and batch_idx == 0:
                log_prediction_visual(
                    image=images[0].detach().cpu().numpy(),
                    mask_true=masks[0].detach().cpu().numpy(),
                    mask_pred=preds[0].detach().cpu().numpy(),
                    step=epoch
                )

        # Averages
        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(val_iou) / len(val_iou)
        avg_pixel_acc = sum(val_pixel_acc) / len(val_pixel_acc)
        avg_precision = sum(val_precision) / len(val_precision)
        avg_recall = sum(val_recall) / len(val_recall)

        # MLflow logs (time-series)
        mlflow.log_metric("val_dice", avg_dice, step=epoch)
        mlflow.log_metric("val_iou", avg_iou, step=epoch)
        mlflow.log_metric("val_pixel_accuracy", avg_pixel_acc, step=epoch)
        mlflow.log_metric("val_precision", avg_precision, step=epoch)
        mlflow.log_metric("val_recall", avg_recall, step=epoch)

    # Extra visuals after the loop
    if log_images and first_batch_cache is not None:
        images, masks, preds, logits = first_batch_cache
        # Grid of first few samples
        log_sample_grid(images, masks, preds, max_n=6, step=epoch)

        # Threshold sweep figure
        if do_threshold_sweep:
            from src.mlflow_utils import log_threshold_sweep
            ths, dvals, ivals = _threshold_sweep_for_batch(logits.to(device), masks.to(device), device)
            log_threshold_sweep(ths, dvals, ivals, out_path=f"figures/threshold_sweep_epoch_{epoch}.png")

    return {
        "dice": avg_dice,
        "iou": avg_iou,
        "pixel_acc": avg_pixel_acc,
        "precision": avg_precision,
        "recall": avg_recall
    }
