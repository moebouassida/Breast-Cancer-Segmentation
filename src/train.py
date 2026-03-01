"""
train.py — Full training pipeline with MLflow + W&B tracking.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --lr 5e-4 --batch-size 16
    python src/train.py --resume checkpoints/best.pt
"""
import argparse
import os
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import UNet, count_parameters
from src.validate import validate
from Data.data_loader import get_dataloaders


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, epoch: int, best_val: float):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path: str, model, optimizer, device: str):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    # Strip DataParallel prefix if present
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", -1) + 1
    best_val = ckpt.get("best_val", float("-inf"))
    print(f"  [ckpt] resumed ← {path}  (epoch {start_epoch}, best Dice {best_val:.4f})")
    return start_epoch, best_val


# ── Learning Curve Logger ─────────────────────────────────────────────────────

def log_learning_curves(history: dict, path: str = "figures/learning_curves.png"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(len(history["train_loss"]))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tomato")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCEWithLogits Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for metric, color in [("val_dice", "steelblue"), ("val_iou", "seagreen"),
                          ("val_precision", "darkorange"), ("val_recall", "purple")]:
        if metric in history:
            ax2.plot(epochs, history[metric], label=metric.replace("val_", ""), color=color)
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="figures")

    if wandb.run is not None:
        wandb.log({"train/learning_curves": wandb.Image(path)})


# ── Main Training Function ────────────────────────────────────────────────────

def train(cfg: Config):
    print(f"\n{'='*60}")
    print(f"  Breast Ultrasound Segmentation — Training")
    print(f"{'='*60}")
    print(f"  Device   : {cfg.device}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Batch    : {cfg.batch_size}")
    print(f"  LR       : {cfg.learning_rate}")
    print(f"  Img size : {cfg.img_size}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=1).to(cfg.device)
    num_params = count_parameters(model)
    print(f"  UNet parameters: {num_params:,}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5, verbose=True
    )

    # ── Resume ────────────────────────────────────────────────────
    start_epoch, best_val = 0, float("-inf")
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        start_epoch, best_val = load_checkpoint(
            cfg.resume_from, model, optimizer, cfg.device
        )

    # ── W&B Init ──────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=f"UNet_bs{cfg.batch_size}_lr{cfg.learning_rate}",
        config={
            "model": "UNet",
            "dataset": cfg.dataset_name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "img_size": cfg.img_size,
            "num_params": num_params,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "loss": "BCEWithLogitsLoss",
        },
    )
    wandb.watch(model, log="gradients", log_freq=50)

    # ── MLflow Init ───────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    history = {
        "train_loss": [], "val_dice": [], "val_iou": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
    }

    with mlflow.start_run(run_name=f"UNet_bs{cfg.batch_size}_lr{cfg.learning_rate}"):
        # Log params to MLflow
        mlflow.set_tags({"model": "UNet", "dataset": cfg.dataset_name})
        mlflow.log_params({
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "img_size": cfg.img_size,
            "num_params": num_params,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        })

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # ── Training Loop ─────────────────────────────────────────
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(cfg.device), masks.to(cfg.device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            current_lr = optimizer.param_groups[0]["lr"]

            # Validate
            metrics = validate(
                model, val_loader, cfg.device,
                epoch=epoch,
                log_images=(epoch % cfg.save_every == 0),
                use_wandb=True,
            )
            val_dice = metrics["dice"]

            # Scheduler step
            scheduler.step(val_dice)

            # Console output
            print(
                f"  Epoch {epoch+1:3d}/{cfg.epochs} | "
                f"loss {avg_loss:.4f} | "
                f"dice {metrics['dice']:.4f} | "
                f"iou {metrics['iou']:.4f} | "
                f"lr {current_lr:.2e}"
            )

            # Log to MLflow + W&B
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            if wandb.run is not None:
                wandb.log(
                    {"train/loss": avg_loss, "train/lr": current_lr}, step=epoch
                )

            # Update history
            history["train_loss"].append(avg_loss)
            history["val_dice"].append(metrics["dice"])
            history["val_iou"].append(metrics["iou"])
            history["val_precision"].append(metrics["precision"])
            history["val_recall"].append(metrics["recall"])
            history["val_f1"].append(metrics["f1"])

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch}.pt")
                save_checkpoint(ckpt_path, model, optimizer, epoch, best_val)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                log_learning_curves(history)

            # Best checkpoint
            if val_dice > best_val:
                best_val = val_dice
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, best_val)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")
                print(f"  ✅ New best Dice: {best_val:.4f} — saved to {best_path}")

                if wandb.run is not None:
                    wandb.run.summary["best_val_dice"] = best_val
                    wandb.run.summary["best_epoch"] = epoch

        # ── Final Artifacts ───────────────────────────────────────
        log_learning_curves(history, "figures/learning_curves_final.png")

        # Save per-epoch metrics CSV
        import pandas as pd
        os.makedirs("metrics", exist_ok=True)
        pd.DataFrame({"epoch": range(len(history["train_loss"])), **history}).to_csv(
            "metrics/per_epoch.csv", index=False
        )
        mlflow.log_artifact("metrics/per_epoch.csv")

        # Save summary JSON (used by evaluate.py)
        summary = {
            "best_val_dice": best_val,
            "final_epoch": cfg.epochs - 1,
            "num_params": num_params,
            "dataset": cfg.dataset_name,
        }
        with open("metrics/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("metrics/summary.json")

        # Log final model to MLflow registry
        example = torch.randn(1, 1, cfg.img_size, cfg.img_size).to(cfg.device)
        mlflow.pytorch.log_model(
            model, "model",
            registered_model_name=cfg.model_name,
            input_example=example,
        )

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best Dice: {best_val:.4f}")
        print(f"  Checkpoint: {os.path.join(cfg.checkpoint_dir, 'best.pt')}")
        print(f"{'='*60}\n")

    wandb.finish()
    return best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train breast ultrasound segmentation model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:      cfg.epochs = args.epochs
    if args.lr:          cfg.learning_rate = args.lr
    if args.batch_size:  cfg.batch_size = args.batch_size
    if args.img_size:    cfg.img_size = args.img_size
    if args.resume:      cfg.resume_from = args.resume

    train(cfg)