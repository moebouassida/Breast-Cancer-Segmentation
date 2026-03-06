"""
HuggingFace Spaces — Breast Cancer Segmentation demo.
Three tabs: Predict · Explain (XAI) · GDPR
"""

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import UNet

API_URL = os.getenv("API_URL", "")
IMG_SIZE = 128
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("checkpoints/best.pt")

transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])


def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[HF Space] Model loaded from {CHECKPOINT_PATH}")
    else:
        print("[HF Space] No checkpoint — demo mode")
    model.eval()
    return model


model = load_model()


def _make_overlay(
    gray01: np.ndarray, mask01: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask01 * alpha, 0, 1)
    return (overlay * 255).astype(np.uint8)


@torch.no_grad()
def _run_predict(image: Image.Image):
    if image is None:
        return None, None, "Please upload an image."
    image_gray = image.convert("L")
    img_t = transform(image_gray).unsqueeze(0).to(DEVICE)
    mask = (torch.sigmoid(model(img_t)) > THRESHOLD).float().squeeze().cpu().numpy()
    gray01 = img_t.squeeze().cpu().numpy()
    coverage = mask.mean() * 100
    return (
        (mask * 255).astype(np.uint8),
        _make_overlay(gray01, mask),
        f"Lesion coverage: {coverage:.1f}%",
    )


def _run_explain(image: Image.Image):
    if image is None:
        return None, None, None, "Please upload an image."
    try:
        from medical_middleware.xai import GradCAM
        from medical_middleware.xai.visualization import overlay_heatmap

        image_gray = image.convert("L")
        img_t = transform(image_gray).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mask = (
                (torch.sigmoid(model(img_t)) > THRESHOLD)
                .float()
                .squeeze()
                .cpu()
                .numpy()
            )
        gray01 = img_t.squeeze().cpu().numpy()
        overlay = _make_overlay(gray01, mask)

        target_layer = None
        for name in ["encoder4", "encoder3", "down4", "down3"]:
            if hasattr(model, name):
                target_layer = getattr(model, name)
                break
        if target_layer is None:
            layers = list(model.children())
            target_layer = layers[len(layers) // 2]

        cam = GradCAM(model, target_layer)
        cam_result = cam.explain(
            img_t, original_image=image, return_base64=False, alpha=0.5
        )
        cam.remove_hooks()

        heatmap_pil = overlay_heatmap(
            image, cam_result["heatmap_raw"], alpha=0.5, return_base64=False
        )
        coverage = mask.mean() * 100
        info = f"Lesion coverage: {coverage:.1f}% | Grad-CAM from last encoder layer"

        return (mask * 255).astype(np.uint8), overlay, np.array(heatmap_pil), info

    except Exception as e:
        mask_img, overlay_img, _ = _run_predict(image)
        return mask_img, overlay_img, None, f"Segmentation OK | XAI error: {str(e)}"


# ── Tab functions ──────────────────────────────────────────────
def tab_predict(image, consent):
    if not consent:
        return None, None, "⚠️ Please check the consent box to proceed."
    mask, overlay, info = _run_predict(image)
    return mask, overlay, info


def tab_explain(image, consent):
    if not consent:
        return None, None, None, "⚠️ Please check the consent box to proceed."
    return _run_explain(image)


def tab_erase(request_id):
    if not request_id or not request_id.strip():
        return "⚠️ Please enter a request ID."
    if not API_URL:
        return "ℹ️ Set the API_URL environment variable to enable erasure."
    try:
        resp = requests.delete(f"{API_URL}/gdpr/erase/{request_id.strip()}", timeout=10)
        data = resp.json()
        if data.get("erased"):
            return (
                "✅ All data associated with this request has been permanently erased."
            )
        return f"ℹ️ {data.get('reason', 'Request ID not found.')}"
    except Exception as e:
        return f"❌ Error contacting API: {e}"


# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(title="🩺 Breast Cancer Segmentation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 🩺 Breast Ultrasound Segmentation
    **U-Net** trained on BUSI dataset — upload a breast ultrasound image to get an AI segmentation mask.
    """
    )

    with gr.Tabs():

        with gr.TabItem("🩺 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    p_image = gr.Image(type="pil", label="Breast Ultrasound Image")
                    p_consent = gr.Checkbox(
                        label="✅ I consent to my image being processed for AI analysis (GDPR Art. 9)",
                        value=False,
                    )
                    p_btn = gr.Button("Segment", variant="primary")
                with gr.Column(scale=1):
                    p_mask = gr.Image(type="numpy", label="Segmentation Mask")
                    p_overlay = gr.Image(type="numpy", label="Overlay (red = lesion)")
                    p_info = gr.Textbox(label="Analysis", interactive=False)
            p_btn.click(
                fn=tab_predict,
                inputs=[p_image, p_consent],
                outputs=[p_mask, p_overlay, p_info],
            )

        with gr.TabItem("🔍 Explain (XAI)"):
            gr.Markdown(
                """
            ### Grad-CAM Explanation
            See *which image regions* the U-Net focused on when deciding where the lesion is.
            **Red = high importance · Blue = low importance**
            """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    e_image = gr.Image(type="pil", label="Breast Ultrasound Image")
                    e_consent = gr.Checkbox(
                        label="✅ I consent to my image being processed for AI analysis (GDPR Art. 9)",
                        value=False,
                    )
                    e_btn = gr.Button("Segment + Explain", variant="primary")
                with gr.Column(scale=2):
                    e_mask = gr.Image(type="numpy", label="Segmentation Mask")
                    e_overlay = gr.Image(type="numpy", label="Segmentation Overlay")
                    e_heatmap = gr.Image(type="numpy", label="Grad-CAM Heatmap")
                    e_info = gr.Textbox(label="Analysis", interactive=False)
            e_btn.click(
                fn=tab_explain,
                inputs=[e_image, e_consent],
                outputs=[e_mask, e_overlay, e_heatmap, e_info],
            )
            gr.Markdown(
                """
            > **How to read Grad-CAM:** Red regions most strongly activated the network's segmentation decision.
            > Clinicians can verify the model is attending to the actual lesion rather than image artifacts.
            """
            )

        with gr.TabItem("🔒 GDPR & Privacy"):
            gr.Markdown(
                """
            ### Your Data Rights (GDPR)

            | Data | Retention | Storage |
            |------|-----------|---------|
            | Uploaded images | **24 hours** (AWS S3 auto-delete) | Anonymized — EXIF stripped |
            | Audit logs | **90 days** | Anonymized IP only |
            | Predictions | **Not stored** | Returned in response only |

            #### Right to Erasure (Article 17)
            Enter your **Request ID** (returned in each API response) to permanently delete all associated data.
            """
            )
            with gr.Row():
                g_id = gr.Textbox(
                    label="Request ID",
                    placeholder="550e8400-e29b-41d4-a716-446655440000",
                )
                g_btn = gr.Button("🗑️ Erase My Data", variant="stop")
            g_out = gr.Textbox(label="Result", interactive=False)
            g_btn.click(fn=tab_erase, inputs=[g_id], outputs=[g_out])
            gr.Markdown(
                """
            ---
            **Data Controller:** Moez Bouassida · **Legal basis:** Art. 9(2)(j) GDPR (scientific research)
            **Framework:** GDPR (EU) 2016/679 · **Contact:** privacy@example.com
            """
            )

    gr.Markdown(
        """
    ---
    **Model:** U-Net (PyTorch) · **Dataset:** BUSI (780 images) · **Author:** [Moez Bouassida](https://github.com/moebouassida)
    """
    )

if __name__ == "__main__":
    demo.launch()
