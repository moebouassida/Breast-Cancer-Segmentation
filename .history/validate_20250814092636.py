from metrics import * 
import torch
from utils import log_prediction_visual

def validate(model, val_loader, device, log_images=False, epoch=None):
    model.eval()
    dices = []
    val_iou = []
    val_pixel_acc = []
    val_precision = []
    val_recall = []

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

    return sum(dices) / len(dices)