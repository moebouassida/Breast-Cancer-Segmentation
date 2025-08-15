import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.config import Config
from Data.data_loader import get_dataloaders
from src.model import UNet
from src.validate import validate
from src.mlflow_utils import log_learning_curves, log_metrics_table, log_summary_json


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _strip_dataparallel_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def save_checkpoint(path, model, optimizer, epoch, best_val):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val,
    }
    _ensure_dir(os.path.dirname(path) or ".")
    torch.save(ckpt, path)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(_strip_dataparallel_prefix(ckpt["model_state"]))
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt.get("epoch", -1) + 1
    best_val = ckpt.get("best_val", float("-inf"))
    print(f"[ckpt] loaded ← {path} (resume at epoch {start_epoch}, best {best_val:.4f})")
    return start_epoch, best_val


def train():
    cfg = Config()

    # MLflow setup
    mlflow.set_tracking_uri("file:Experiments/mlruns")
    mlflow.set_experiment(cfg.mlflow_experiment)

    device = cfg.device
    print("Getting dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(cfg)
    print("Starting training...")

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    _ensure_dir(cfg.checkpoint_dir)
    start_epoch = 0
    best_val = float("-inf")
    if getattr(cfg, "resume_from", None) and os.path.isfile(cfg.resume_from):
        start_epoch, best_val = load_checkpoint(cfg.resume_from, model, optimizer, device)

    run_name = f"UNet_bs{cfg.batch_size}_lr{cfg.learning_rate}"
    with mlflow.start_run(run_name=run_name):
        # Tags & params
        mlflow.set_tag("model", "UNet")
        mlflow.set_tag("notes", getattr(cfg, "notes", ""))
        if hasattr(cfg, "dataset_name"):
            mlflow.set_tag("dataset", cfg.dataset_name)

        mlflow.log_params({
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "save_every": cfg.save_every,
            "img_size": getattr(cfg, "img_size", "unknown"),
        })

        # Track history for custom curves/tables
        history = {
            "train_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_pixel_accuracy": [],
            "val_precision": [],
            "val_recall": [],
        }

        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            metrics = validate(model, val_loader, device, log_images=True, epoch=epoch)
            val_dice = metrics["dice"]

            print(f"Epoch {epoch+1}/{cfg.epochs} | loss {avg_loss:.4f} | "
                  f"val_dice {metrics['dice']:.4f} | val_iou {metrics['iou']:.4f}")

            # Time-series logs for MLflow default charts
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k if k!='pixel_acc' else 'pixel_accuracy'}", v, step=epoch)

            # Update history for custom figures/tables
            history["train_loss"].append(avg_loss)
            history["val_dice"].append(metrics["dice"])
            history["val_iou"].append(metrics["iou"])
            history["val_pixel_accuracy"].append(metrics["pixel_acc"])
            history["val_precision"].append(metrics["precision"])
            history["val_recall"].append(metrics["recall"])

            # Periodic checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                last_path = os.path.join(cfg.checkpoint_dir, f"{epoch}.pt")
                save_checkpoint(last_path, model, optimizer, epoch, best_val)
                mlflow.log_artifact(last_path, artifact_path="checkpoints")

            # Best checkpoint
            if val_dice > best_val:
                best_val = val_dice
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, best_val)
                mlflow.log_artifact(best_path, artifact_path="checkpoints")

            # Periodically log learning curves so far
            if (epoch + 1) % max(1, (cfg.save_every // 2)) == 0:
                log_learning_curves(history, out_path="figures/learning_curves.png")

        # Final logs/artifacts
        log_learning_curves(history, out_path="figures/learning_curves_final.png")
        log_metrics_table(history, out_csv="metrics/per_epoch.csv")

        summary = {
            "best_val_dice": best_val,
            "final_epoch": cfg.epochs - 1,
            "num_params": int(sum(p.numel() for p in model.parameters())),
        }
        log_summary_json(summary, out_json="summary.json")

        # Log the final model (with example for signature)
        example = torch.randn(1, 1, getattr(cfg, "img_size", 128), getattr(cfg, "img_size", 128)).to(device)
        mlflow.pytorch.log_model(model, "model", input_example=example)

        # Also log any accumulated prediction images folder (if present)
        if os.path.isdir("temp_vis"):
            mlflow.log_artifacts("temp_vis", artifact_path="predictions")


if __name__ == "__main__":
    train()
