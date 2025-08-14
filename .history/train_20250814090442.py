import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from config import Config
from data_loader import get_dataloaders
from model import UNet
from metrics import dice_score
from utils import log_prediction_visual

def train():
    cfg = Config()
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.mlflow_experiment)

    device = cfg.device
    train_loader, val_loader, _ = get_dataloaders(cfg)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate
        })

        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_dice = validate(model, val_loader, device, log_images=True, epoch=epoch)
            print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {avg_loss:.4f}, Val Dice: {val_dice:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)

        mlflow.pytorch.log_model(model, "model")

def validate(model, val_loader, device, log_images=False, epoch=None):
    model.eval()
    dices = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))
            preds = (outputs > 0.5).float()
            dices.append(dice_score(preds, masks).item())

            # Log just the first batch images for visualization
            if log_images and batch_idx == 0:
                log_prediction_visual(
                    image=images[0].cpu().numpy(),
                    mask_true=masks[0].cpu().numpy(),
                    mask_pred=preds[0].cpu().numpy(),
                    step=epoch
                )

    return sum(dices) / len(dices)


if __name__ == "__main__":
    train()
