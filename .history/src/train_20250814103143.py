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

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _strip_dataparallel_prefix(state_dict):
    # Handles checkpoints saved with nn.DataParallel
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
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.mlflow_experiment)

    device = cfg.device
    print("Getting dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(cfg)
    print("Starting training...")

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # --- resume logic ---
    _ensure_dir(cfg.checkpoint_dir)
    start_epoch = 0
    best_val = float("-inf")
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        start_epoch, best_val = load_checkpoint(cfg.resume_from, model, optimizer, device)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate
        })

        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            total_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            metrics = validate(model, val_loader, device, log_images=True, epoch=epoch)
            val_dice = metrics["dice"]

            print(f"Epoch {epoch+1}/{cfg.epochs} | loss {avg_loss:.4f} | val_dice {val_dice:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)

            # --- save checkpoints ---
            if (epoch + 1) % cfg.save_every == 0:
                last_path = os.path.join(cfg.checkpoint_dir, "last.pt")
                save_checkpoint(last_path, model, optimizer, epoch, best_val)
                # you can also log it to mlflow if desired
                mlflow.log_artifact(last_path)

            if val_dice > best_val:
                best_val = val_dice
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, best_val)
                mlflow.log_artifact(best_path)

        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
