import matplotlib.pyplot as plt
import os

def log_prediction_visual(image, mask_true, mask_pred, step):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(mask_true.squeeze(), cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(mask_pred.squeeze(), cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    plt.tight_layout()

    os.makedirs("mlflow_preds", exist_ok=True)
    file_path = f"mlflow_preds/epoch_{step}.png"
    plt.savefig(file_path)
    plt.close(fig)

    mlflow.log_artifact(file_path, artifact_path="predictions")
