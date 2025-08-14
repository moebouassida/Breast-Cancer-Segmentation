import mlflow.pytorch
import torch

class ModelWrapper:
    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.model = mlflow.pytorch.load_model(f"models:/{model_name}/Production").to(device)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            output = torch.sigmoid(self.model(image_tensor.to(self.device)))
            return (output > 0.5).float()
