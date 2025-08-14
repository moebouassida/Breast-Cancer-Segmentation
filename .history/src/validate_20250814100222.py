import torch
import mlflow
from .metrics import * 
from src.utils import log_prediction_visual

def validate(model, val_loader, device, log_images=False, epoch=None):
    model = model.to(device)
    model.eval()
    
    dices, val_iou, val_pixel_acc, val_precision, val_recall = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()
            
            dices.append(dice_score(preds, masks).item())
            val_iou.append(iou_score(preds, masks).item())
            val_pixel_acc.append(pixel_accuracy(preds, masks).item())
            val_precision.append(precision_score(preds, masks).item())
            val_recall.append(recall_score(preds, masks).item())

            # Log just the first batch images for visualization
            if log_images and batch_idx == 0:
                log_prediction_visual(
                    image=images[0].cpu().numpy(),
                    mask_true=masks[0].cpu().numpy(),
                    mask_pred=preds[0].cpu().numpy(),
                    step=epoch
                )
        
        # Average metrics        
        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(val_iou) / len(val_iou)
        avg_pixel_acc = sum(val_pixel_acc) / len(val_pixel_acc)
        avg_precision = sum(val_precision) / len(val_precision)
        avg_recall = sum(val_recall) / len(val_recall)
        
        # MLflow logs
        mlflow.log_metric("val_dice", avg_dice, step=epoch)
        mlflow.log_metric("val_iou", avg_iou, step=epoch)
        mlflow.log_metric("val_pixel_accuracy", avg_pixel_acc, step=epoch)
        mlflow.log_metric("val_precision", avg_precision, step=epoch)
        mlflow.log_metric("val_recall", avg_recall, step=epoch)
        
    return {
    "dice": avg_dice,
    "iou": avg_iou,
    "pixel_acc": avg_pixel_acc,
    "precision": avg_precision,
    "recall": avg_recall
} 