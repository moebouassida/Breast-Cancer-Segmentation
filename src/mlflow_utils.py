import os
import json
import mlflow
import matplotlib.pyplot as plt
import pandas as pd


def log_learning_curves(history: dict, out_path="figures/learning_curves.png"):
    """
    history: dict of lists keyed by metric names, all same length.
    Example keys: train_loss, val_dice, val_iou, val_precision, val_recall
    """
    if not history:
        return

    # Plot each list against epoch index
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(len(next(iter(history.values()))))
    for name, vals in history.items():
        if len(vals) == len(list(epochs)):
            ax.plot(epochs, vals, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training / Validation Curves")
    ax.legend(loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlflow.log_figure(fig, out_path)
    plt.close(fig)


def log_threshold_sweep(thresholds, dice_vals, iou_vals, out_path="figures/threshold_sweep.png"):
    """Plot dice & iou vs threshold."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, dice_vals, label="dice")
    ax.plot(thresholds, iou_vals, label="iou")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title("Threshold Sweep")
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mlflow.log_figure(fig, out_path)
    plt.close(fig)


def log_metrics_table(history: dict, out_csv="metrics/per_epoch.csv"):
    """
    Save a per-epoch CSV from history dict.
    """
    if not history:
        return
    length = len(next(iter(history.values())))
    data = {"epoch": list(range(length))}
    for k, v in history.items():
        if len(v) == length:
            data[k] = v
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    mlflow.log_artifact(out_csv)


def log_summary_json(summary: dict, out_json="summary.json"):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    mlflow.log_artifact(out_json)
