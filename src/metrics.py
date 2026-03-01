import torch


EPS = 1e-7


def dice_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice coefficient — primary segmentation metric. Range [0, 1]."""
    preds = preds.float().contiguous().view(preds.size(0), -1)
    targets = targets.float().contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * intersection + EPS) / (union + EPS)).mean()


def iou_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Intersection over Union (Jaccard index). Range [0, 1]."""
    preds = preds.float().contiguous().view(preds.size(0), -1)
    targets = targets.float().contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + EPS) / (union + EPS)).mean()


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fraction of correctly classified pixels."""
    return (preds == targets).float().mean()


def precision_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Precision = TP / (TP + FP)."""
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()
    return (tp + EPS) / (tp + fp + EPS)


def recall_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Recall / Sensitivity = TP / (TP + FN). Critical for medical AI."""
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fn = ((preds == 0) & (targets == 1)).float().sum()
    return (tp + EPS) / (tp + fn + EPS)


def f1_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """F1 = harmonic mean of precision and recall."""
    p = precision_score(preds, targets)
    r = recall_score(preds, targets)
    return (2 * p * r + EPS) / (p + r + EPS)


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute all metrics in one call. Returns a plain dict of floats."""
    return {
        "dice": dice_score(preds, targets).item(),
        "iou": iou_score(preds, targets).item(),
        "pixel_accuracy": pixel_accuracy(preds, targets).item(),
        "precision": precision_score(preds, targets).item(),
        "recall": recall_score(preds, targets).item(),
        "f1": f1_score(preds, targets).item(),
    }
