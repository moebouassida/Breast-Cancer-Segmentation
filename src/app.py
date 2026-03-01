"""
app.py — FastAPI + Gradio inference server.

Endpoints:
    GET  /health       — liveness check
    POST /predict      — upload image → segmentation mask + overlay
    GET  /gradio       — interactive Gradio demo
"""
import io
import os
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.model import UNet

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "best.pt"
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CKPT)))
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Ultrasound Segmentation API",
    description="U-Net segmentation of breast ultrasound images. "
                "Trained on the BUSI dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class ModelWrapper:
    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.device = device
        self.model = UNet(in_channels=1, out_channels=1).to(device)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Train the model first or set CHECKPOINT_PATH env var."
            )

        ckpt = torch.load(str(checkpoint_path), map_location=device)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing}")
        self.model.eval()
        print(f"[model] loaded from {checkpoint_path} on {device}")

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = self.model(tensor.to(self.device))
        return (torch.sigmoid(logits) > threshold).float()


_model: Optional[ModelWrapper] = None


def get_model() -> ModelWrapper:
    global _model
    if _model is None:
        _model = ModelWrapper(CHECKPOINT_PATH, device=DEVICE)
    return _model


# ── Transforms ────────────────────────────────────────────────────────────────
transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_b64_png(arr: np.ndarray) -> str:
    """Convert a numpy array (H,W) or (H,W,3) to a base64 PNG string."""
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    mode = "RGB" if arr.ndim == 3 else "L"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_overlay(gray01: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Red overlay of segmentation mask on grayscale image."""
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask01 * alpha, 0, 1)
    return overlay


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "threshold": THRESHOLD,
        "model_loaded": CHECKPOINT_PATH.exists(),
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Grayscale ultrasound image"),
    return_images: bool = True,
):
    """
    Segment a breast ultrasound image.

    Returns:
    - `prediction_mask`: raw 2D mask as nested list
    - `mask_png_b64`: PNG of the binary mask (base64)
    - `overlay_png_b64`: PNG of the mask overlaid on the input (base64)
    """
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_t = transform(image).unsqueeze(0)  # (1, 1, H, W)
    pred = get_model().predict(img_t, threshold=THRESHOLD)
    mask = pred.squeeze().cpu().numpy().astype(np.float32)  # (H, W)

    result = {
        "prediction_mask": mask.tolist(),
        "device_used": DEVICE,
        "threshold": THRESHOLD,
        "img_size": IMG_SIZE,
    }

    if return_images:
        gray01 = img_t.squeeze().cpu().numpy()
        result["mask_png_b64"] = to_b64_png(mask)
        result["overlay_png_b64"] = to_b64_png(make_overlay(gray01, mask))

    return result


# ── Gradio UI ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def gradio_predict(image: Image.Image):
    if image is None:
        return None, None

    image = image.convert("L")
    img_t = transform(image).unsqueeze(0)
    pred = get_model().predict(img_t, threshold=THRESHOLD)
    mask = pred.squeeze().cpu().numpy().astype(np.float32)
    gray01 = img_t.squeeze().cpu().numpy()

    mask_display = cv2.resize(
        (mask * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
    )
    overlay_np = make_overlay(gray01, mask)
    overlay_display = cv2.resize(
        (overlay_np * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR
    )
    return mask_display, overlay_display


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Breast Ultrasound Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Segmentation Mask"),
        gr.Image(type="numpy", label="Overlay (red = lesion)"),
    ],
    title="🩺 Breast Ultrasound Segmentation",
    description=(
        "Upload a breast ultrasound image to get an AI-generated segmentation mask. "
        "Trained on the BUSI dataset using a custom U-Net architecture."
    ),
    examples=[],
    allow_flagging="never",
)

gr.mount_gradio_app(app, demo, path="/gradio")