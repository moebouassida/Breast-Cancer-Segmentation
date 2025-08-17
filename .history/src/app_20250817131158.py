# app.py
import os
import io
import base64
from typing import Optional

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import gradio as gr

# --------- Config ---------
CHECKPOINT_PATH = r"C:/Users/moez/Breast-Cancer-Segmentation/Experiments/mlruns/477266664753470372/b4d44cc469c644f19397dfac844c1491/artifacts/checkpoints/best.pt"
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
from model import UNet  # make sure UNet class is imported

class ModelWrapper:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model = UNet(in_channels=1, out_channels=1).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict_mask(self, image_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        logits = self.model(image_tensor.to(self.device))
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()  # (B,1,H,W)

_model: Optional[ModelWrapper] = None
def get_model() -> ModelWrapper:
    global _model
    if _model is None:
        _model = ModelWrapper(CHECKPOINT_PATH, device=DEVICE)
    return _model

# --------- Transforms ---------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# --------- Utils ---------
def mask_to_b64(mask_01: np.ndarray) -> str:
    m = (mask_01 * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(m, "L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def overlay_to_b64(gray01: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> str:
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask01 * alpha, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((overlay * 255).astype(np.uint8), "RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# --------- API Endpoints ---------
@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE, "img_size": IMG_SIZE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), return_images: bool = True):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_t = transform(image).unsqueeze(0)  # (1,1,H,W)
    model = get_model()
    pred = model.predict_mask(img_t, threshold=THRESHOLD)
    mask = pred.squeeze().cpu().numpy().astype(np.float32)  # (H,W)

    result = {
        "prediction_mask": mask.tolist(),
        "device_used": DEVICE,
        "threshold": THRESHOLD,
        "img_size": IMG_SIZE,
    }
    if return_images:
        gray01 = img_t.squeeze().cpu().numpy()
        result["mask_png_b64"] = mask_to_b64(mask)
        result["overlay_png_b64"] = overlay_to_b64(gray01, mask)
    return result

# --------- Gradio UI  ---------
@torch.no_grad()
def gradio_predict(image: Image.Image):
    image = image.convert("L")
    img_t = transform(image).unsqueeze(0)
    model = get_model()
    pred = model.predict_mask(img_t, threshold=THRESHOLD)
    mask = pred.squeeze().cpu().numpy().astype(np.float32)

    gray01 = img_t.squeeze().cpu().numpy()
    overlay = np.stack([gray01, gray01, gray01], axis=-1)
    overlay[..., 0] = np.maximum(overlay[..., 0], mask * 0.5)
    overlay = (overlay * 255).astype(np.uint8)

    mask_img = (mask * 255).astype(np.uint8)
    return mask_img, overlay

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Ultrasound Image"),
    outputs=[
        gr.Image(type="numpy", label="Predicted Mask"),
        gr.Image(type="numpy", label="Overlay"),
    ],
    title="Breast Ultrasound Segmentation",
    description="FastAPI backend with Gradio UI. Model loaded from local checkpoint.",
)

# Mount Gradio under FastAPI
gr.mount_gradio_app(app, demo, path="/gradio")
