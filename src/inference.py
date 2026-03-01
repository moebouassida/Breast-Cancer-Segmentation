"""
inference.py — Load model from MLflow Model Registry for production inference.

Usage:
    from src.inference import SegmentationInference
    inf = SegmentationInference()
    mask = inf.predict("path/to/image.png")
"""
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T


class SegmentationInference:
    """
    Production inference wrapper.
    Loads model from MLflow Model Registry (preferred) or local checkpoint.
    """

    def __init__(
        self,
        model_name: str = "BreastSeg",
        stage: str = "Production",
        checkpoint_path: str = None,
        device: str = None,
        img_size: int = 128,
        threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.threshold = threshold
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

        self.model = self._load_model(model_name, stage, checkpoint_path)
        self.model.eval()

    def _load_model(self, model_name, stage, checkpoint_path):
        # Try MLflow registry first
        if os.getenv("MLFLOW_TRACKING_URI") or os.path.exists("Experiments/mlruns"):
            try:
                import mlflow.pytorch
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:Experiments/mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                model = mlflow.pytorch.load_model(
                    f"models:/{model_name}/{stage}"
                ).to(self.device)
                print(f"[inference] Loaded from MLflow registry: {model_name}/{stage}")
                return model
            except Exception as e:
                print(f"[inference] MLflow registry unavailable ({e}), falling back to checkpoint")

        # Fallback to local checkpoint
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/best.pt"

        from src.model import UNet
        model = UNet(in_channels=1, out_channels=1).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[inference] Loaded from checkpoint: {checkpoint_path}")
        return model

    @torch.no_grad()
    def predict(self, image_input, return_overlay: bool = False):
        """
        Predict segmentation mask for a single image.

        Args:
            image_input: path string, PIL Image, or numpy array
            return_overlay: if True, also return colored overlay

        Returns:
            mask (np.ndarray H×W, values 0 or 1)
            overlay (np.ndarray H×W×3, float32) — only if return_overlay=True
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("L")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("L")
        else:
            image = image_input.convert("L")

        orig_size = image.size  # (W, H)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.sigmoid(logits)
        mask_small = (probs > self.threshold).float().squeeze().cpu().numpy()

        # Resize back to original resolution
        from PIL import Image as PILImage
        mask = np.array(
            PILImage.fromarray((mask_small * 255).astype(np.uint8)).resize(
                orig_size, PILImage.NEAREST
            )
        ) / 255.0

        if not return_overlay:
            return mask

        gray = np.array(image.resize(orig_size)) / 255.0
        overlay = np.stack([gray, gray, gray], axis=-1)
        overlay[..., 0] = np.clip(overlay[..., 0] + mask * 0.5, 0, 1)
        return mask, overlay