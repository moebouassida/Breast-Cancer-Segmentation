# app.py
import os
import io
import base64
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import mlflow
import mlflow.pytorch

# --------- Config via env ---------
MODEL_NAME = os.getenv("MODEL_NAME", "BreastSeg")  # MLflow Registered Model name
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:Experiments/mlruns")
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- FastAPI ---------
app = FastAPI(title="Breast Ultrasound Segmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# --------- Model Wrapper ---------
class ModelWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.pytorch.load_model(f"models:/{model_name}/Production").to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, image_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.model(image_tensor.to(self.device))
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()  # (B,1,H,W)

_model_wrapper: Optional[ModelWrapper] = None

def get_model() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        try:
            _model_wrapper = ModelWrapper(MODEL_NAME, device=DEVICE)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MLflow model 'models:/{MODEL_NAME}/Production' "
                f"from '{MLFLOW_TRACKING_URI}'. Error: {e}"
            )
    return _model_wrapper

# --------- Transforms ---------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),    
])

# --------- Utils ---------
def mask_to_base64_png(mask_01: np.ndarray) -> str:
    """ mask_01: (H, W) float32 in {0,1}. Returns base64-encoded PNG bytes. """
    from PIL import Image
    m = (mask_01 * 255).astype(np.uint8)
    img = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def overlay_base64_png(image_gray: np.ndarray, mask_01: np.ndarray, alpha: float = 0.4) -> str:
    """
    image_gray: (H,W) float32 in [0,1]
    mask_01: (H,W) float32 in {0,1}
    Overlay mask in red. Returns base64 PNG.
    """
    h, w = image_gray.shape
    rgb = np.stack([image_gray, image_gray, image_gray], axis=-1)  # H,W,3
    overlay = rgb.copy()
    # red channel boosted where mask==1
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_01 * alpha, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    pil = Image.fromarray(overlay, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# --------- Endpoints ---------
@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE, "model_name": MODEL_NAME, "img_size": IMG_SIZE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), return_images: bool = True):
    """
    Returns:
      - prediction_mask: 2D list (H x W) of 0/1
      - (optional) mask_png_b64: base64 PNG of mask
      - (optional) overlay_png_b64: base64 PNG overlay of mask on image
    """
    # Read and parse image
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")  # grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Transform
    image_tensor = transform(image).unsqueeze(0)  # (1,1,H,W)
    model = get_model()

    # Predict
    pred_mask = model.predict_mask(image_tensor, threshold=THRESHOLD)  # (1,1,H,W)
    mask = pred_mask.squeeze().cpu().numpy().astype(np.float32)  # (H,W), {0,1}

    result = {
        "prediction_mask": mask.tolist(),
        "device_used": DEVICE,
        "threshold": THRESHOLD,
        "img_size": IMG_SIZE,
    }

    if return_images:
        # Build PNGs
        # Convert original (already resized via transform) to [0,1] gray for overlay
        # (We re-run transform on PIL -> tensor -> numpy)
        img_gray_01 = image_tensor.squeeze().cpu().numpy()  # (H,W) in [0,1]
        result["mask_png_b64"] = mask_to_base64_png(mask)
        result["overlay_png_b64"] = overlay_base64_png(img_gray_01, mask)

    return result
