"""
app.py — FastAPI + Breast Cancer U-Net Segmentation.

Endpoints:
    GET    /health              — liveness check
    POST   /predict             — upload image → mask + overlay
    POST   /explain/predict     — upload image → mask + Grad-CAM heatmap
    GET    /explain/methods     — XAI method info
    GET    /metrics             — Prometheus metrics
    GET    /gdpr/status         — GDPR compliance status
    DELETE /gdpr/erase/{id}     — right to erasure
"""

import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.model import UNet

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ── Config ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = BASE_DIR / "checkpoints" / "best.pt"
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", str(DEFAULT_CKPT)))
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Ultrasound Segmentation API",
    description="U-Net segmentation of breast ultrasound images. Trained on BUSI dataset.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Middleware ─────────────────────────────────────────────────
from medical_middleware import setup_middleware # noqa: E402
from medical_middleware.config import MiddlewareConfig # noqa: E402
from medical_middleware.storage import get_s3_client # noqa: E402
from medical_middleware.storage.retention_s3 import S3RetentionManager # noqa: E402

middleware_cfg = MiddlewareConfig(
    app_name="breast-cancer-api",
    consent_required_paths=["/predict", "/explain/predict"],
    csp_policy="default-src * 'unsafe-inline' 'unsafe-eval' data: blob:",
)
setup_middleware(app, middleware_cfg)

# ── S3 singleton ──────────────────────────────────────────────
s3 = get_s3_client()
retention = S3RetentionManager(s3_client=s3) if s3.available else None


# ── Model ─────────────────────────────────────────────────────
class ModelWrapper:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        self.model = UNet(in_channels=1, out_channels=1).to(device)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict(self, tensor, threshold=0.5):
        return (torch.sigmoid(self.model(tensor.to(self.device))) > threshold).float()


_model_wrapper: Optional[ModelWrapper] = None


def get_model() -> ModelWrapper:
    global _model_wrapper
    if _model_wrapper is None:
        _model_wrapper = ModelWrapper(CHECKPOINT_PATH, DEVICE)
    return _model_wrapper


transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


# ── Startup — preload model ────────────────────────────────────
@app.on_event("startup")
async def startup():
    get_model()


# ── Helpers ───────────────────────────────────────────────────
def to_b64_png(arr: np.ndarray) -> str:
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    mode = "RGB" if arr.ndim == 3 else "L"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_overlay(
    gray01: np.ndarray, mask01: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask01 * alpha, 0, 1)
    return overlay


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": _model_wrapper is not None,
        "checkpoint": str(CHECKPOINT_PATH),
        "s3_enabled": s3.available,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    return_images: bool = True,
):
    """Segment breast ultrasound image → mask + overlay."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Anonymize + upload to S3
    request_id = getattr(request.state, "request_id", None)
    if retention and request_id:
        from medical_middleware.gdpr.anonymizer import ImageAnonymizer

        anon_bytes = ImageAnonymizer.anonymize_bytes(
            content, file.content_type or "image/jpeg"
        )
        retention.register_upload(request_id, anon_bytes, file.filename or "image.jpg")

    img_t = transform(image).unsqueeze(0)
    mask = (
        get_model().predict(img_t, THRESHOLD).squeeze().cpu().numpy().astype(np.float32)
    )

    result = {
        "device_used": DEVICE,
        "threshold": THRESHOLD,
        "request_id": request_id,
        "coverage_pct": round(float(mask.mean() * 100), 2),
    }
    if return_images:
        gray01 = img_t.squeeze().cpu().numpy()
        result["mask_png_b64"] = to_b64_png(mask)
        result["overlay_png_b64"] = to_b64_png(make_overlay(gray01, mask))

    return result


@app.post("/explain/predict")
async def explain_predict(request: Request, file: UploadFile = File(...)):
    """Segment + Grad-CAM explanation → mask + heatmap overlay."""
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_t = transform(image).unsqueeze(0)
    model = get_model()
    gray01 = img_t.squeeze().cpu().numpy()
    mask = model.predict(img_t, THRESHOLD).squeeze().cpu().numpy().astype(np.float32)
    request_id = getattr(request.state, "request_id", None)

    xai_result = {}
    try:
        from medical_middleware.xai import GradCAM

        target_layer = None
        for name in ["encoder4", "encoder3", "down4", "down3"]:
            if hasattr(model.model, name):
                target_layer = getattr(model.model, name)
                break
        if target_layer is None:
            layers = list(model.model.children())
            target_layer = layers[len(layers) // 2]

        cam = GradCAM(model.model, target_layer)
        result = cam.explain(img_t, original_image=image, return_base64=True, alpha=0.5)
        cam.remove_hooks()

        xai_result = {
            "heatmap_b64": result["heatmap_b64"],
            "method": "gradcam",
            "clinical_note": "Red regions indicate areas that most strongly influenced the segmentation decision.",
        }

        # Upload heatmap to S3
        if s3.available and request_id:
            img_bytes = base64.b64decode(result["heatmap_b64"])
            s3.upload_bytes(
                img_bytes,
                key=f"{middleware_cfg.app_name}/xai/{request_id}_gradcam.png",
                bucket=s3.bucket_logs,
                content_type="image/png",
            )

    except Exception as e:
        xai_result["error"] = str(e)

    return {
        "mask_png_b64": to_b64_png(mask),
        "overlay_png_b64": to_b64_png(make_overlay(gray01, mask)),
        "coverage_pct": round(float(mask.mean() * 100), 2),
        "xai": xai_result,
        "request_id": request_id,
    }


@app.get("/explain/methods")
def explain_methods():
    return {
        "model_type": "unet",
        "method": "Grad-CAM",
        "description": "Highlights which image regions drove the U-Net segmentation decision.",
        "reference": "Selvaraju et al. (2017) https://arxiv.org/abs/1610.02391",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
