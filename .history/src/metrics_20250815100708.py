import torch

def dice_score(preds, targets, eps=1e-7):
    # preds, targets: (N, 1, H, W) or same shape
    preds = preds.float().view(preds.size(0), -1)
    targets = targets.float().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()

def iou_score(preds, targets, eps=1e-7):
    preds = preds.float().view(preds.size(0), -1)
    targets = targets.float().view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()

def pixel_accuracy(preds, targets):
    correct = (preds == targets).float().mean()
    return correct

def precision_score(preds, targets, eps=1e-7):
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()
    return (tp + eps) / (tp + fp + eps)

def recall_score(preds, targets, eps=1e-7):
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fn = ((preds == 0) & (targets == 1)).float().sum()
    return (tp + eps) / (tp + fn + eps)
