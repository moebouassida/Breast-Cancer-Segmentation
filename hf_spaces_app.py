"""
HuggingFace Spaces entry point.
This file is the root app.py that HF Spaces expects.
It loads the model from the checkpoint baked into the Space.
"""
import os
import sys
import torch
import numpy as np
import gradio as gr
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import UNet

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 128
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("checkpoints/best.pt")

transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[HF Space] Model loaded from {CHECKPOINT_PATH}")
    else:
        print("[HF Space] No checkpoint found — using untrained model for demo")
    model.eval()
    return model


model = load_model()


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(image: Image.Image):
    if image is None:
        return None, None, "Please upload an image."

    image_gray = image.convert("L")
    img_t = transform(image_gray).unsqueeze(0).to(DEVICE)
    logits = model(img_t)
    probs = torch.sigmoid(logits)
    mask = (probs > THRESHOLD).float().squeeze().cpu().numpy()

    gray01 = img_t.squeeze().cpu().numpy()

    # Mask image
    mask_img = (mask * 255).astype(np.uint8)

    # Overlay: red channel highlights the lesion
    rgb = np.stack([gray01, gray01, gray01], axis=-1)
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask * 0.5, 0, 1)
    overlay_img = (overlay * 255).astype(np.uint8)

    coverage = mask.mean() * 100
    info = f"Lesion coverage: {coverage:.1f}% of image area"

    return mask_img, overlay_img, info


# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Breast Ultrasound Segmentation") as demo:
    gr.Markdown("""
    # 🩺 Breast Ultrasound Segmentation
    AI-powered segmentation of breast ultrasound images using a custom U-Net architecture.
    Trained on the [BUSI dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset).

    **Upload a breast ultrasound image** to get an automatic segmentation mask highlighting lesion regions.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Ultrasound Image")
            submit_btn = gr.Button("Segment", variant="primary")

        with gr.Column():
            mask_output = gr.Image(type="numpy", label="Segmentation Mask")
            overlay_output = gr.Image(type="numpy", label="Overlay (red = lesion)")
            info_output = gr.Textbox(label="Analysis", interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[mask_output, overlay_output, info_output],
    )

    gr.Markdown("""
    ---
    ### About
    - **Model**: Custom U-Net (PyTorch)
    - **Dataset**: BUSI — 780 breast ultrasound images (benign/malignant/normal)
    - **Framework**: PyTorch + FastAPI
    - **Author**: [Moez Bouassida](https://github.com/moebouassida)
    """)

if __name__ == "__main__":
    demo.launch()