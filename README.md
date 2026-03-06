# 🩺 Breast Ultrasound Segmentation

> Semantic segmentation of breast ultrasound images using a custom U-Net architecture.  
> Trained on the BUSI dataset · Served via FastAPI · GDPR-compliant · Deployed with Docker · CI/CD with GitHub Actions.

![CI Pipeline](https://github.com/moebouassida/Breast-Cancer-Segmentation/actions/workflows/ci_cd.yml/badge.svg)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/moebouassida/breast-ultrasound-segmentation)
![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Results

| Metric | Score |
|---|---|
| **Dice Coefficient**     | `0.73` |
| **IoU (Jaccard)**        | `0.66` |
| **Precision**            | `0.80` |
| **Recall (Sensitivity)** | `0.71` |
| **Pixel Accuracy**       | `0.96` |

> Evaluated on the BUSI test set (benign + malignant + normal classes).  
> Training tracked with W&B — [view experiment dashboard](https://wandb.ai/moebouassida-soci-t-g-n-rale/breast-ultrasound-segmentation).

---

## 🎥 Demo

[![HuggingFace Demo](https://img.shields.io/badge/▶%20Try%20Live%20Demo-HuggingFace%20Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/moebouassida/breast-ultrasound-segmentation)

---

## 🏗️ Architecture

```
Input (1×128×128 grayscale ultrasound)
        ↓
   Encoder (4× conv blocks + MaxPool)
   64 → 128 → 256 → 512 channels
        ↓
   Bottleneck (1024 channels)
        ↓
   Decoder (4× ConvTranspose2d + skip connections)
   512 → 256 → 128 → 64 channels
        ↓
Output (1×128×128 binary segmentation mask)
```

Custom U-Net with:
- BatchNorm after every convolution
- Skip connections between encoder and decoder
- BCEWithLogitsLoss for training stability
- Adam optimizer with lr=5e-4

---

## 🛡️ Production Middleware

This project integrates [medical-ai-middleware](https://github.com/moebouassida/medical-ai-middleware) — a production-grade middleware package built for medical AI APIs:

| Feature | Details |
|---|---|
| **GDPR Compliance** | Consent enforcement, right to erasure, audit logging |
| **Data Anonymization** | EXIF stripping, IP anonymization before S3 upload |
| **S3 Storage** | Anonymized uploads auto-deleted after 24h, audit logs after 90 days |
| **Prometheus Metrics** | Request count, latency, error rate at `/metrics` |
| **Rate Limiting** | 10 req/min on `/predict`, 30 req/min default |
| **Security Headers** | HSTS, CSP, X-Frame-Options, X-Request-ID on every response |

---

## 📁 Project Structure

```
Breast-Cancer-Segmentation/
├── .github/
│   └── workflows/
│       └── ci.yml              ← GitHub Actions: lint → test → docker → deploy
├── Data/
│   ├── Dataset_BUSI_with_GT/   ← BUSI dataset (not committed, tracked by DVC)
│   └── data_loader.py
├── Docker/
│   └── Dockerfile              ← Production container
├── src/
│   ├── model.py                ← U-Net architecture
│   ├── train.py                ← Training loop with W&B + MLflow
│   ├── evaluate.py             ← Evaluation + quality gates
│   ├── validate.py             ← Validation loop
│   ├── metrics.py              ← Dice, IoU, precision, recall, F1
│   ├── config.py               ← Hyperparameters and paths
│   ├── mlflow_utils.py         ← Learning curves, threshold sweep
│   ├── utils.py                ← Visualization helpers
│   ├── inference.py            ← MLflow model registry inference
│   └── app.py                  ← FastAPI serving + middleware
├── hf_spaces_app.py            ← Gradio demo (HuggingFace Spaces)
├── tests/
│   ├── test_suite.py           ← Model, metrics, preprocessing tests
│   └── test_api.py             ← API + middleware integration tests
├── checkpoints/                ← Model weights (not committed)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/moebouassida/Breast-Cancer-Segmentation.git
cd Breast-Cancer-Segmentation

docker build -f Docker/Dockerfile -t breast-seg-api .
docker run --rm -p 8000:8000 \
  -e S3_ENABLED=false \
  breast-seg-api
```

Open:
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Prometheus metrics:** http://localhost:8000/metrics

### Option 2 — Local Python

```bash
git clone https://github.com/moebouassida/Breast-Cancer-Segmentation.git
cd Breast-Cancer-Segmentation
python -m venv venv && venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

---

## 🔌 API Reference

All endpoints on `/predict` and `/explain/predict` require the consent header:
```
X-Data-Consent: true
```

### GET `/health`
```json
{
  "status": "healthy",
  "metrics": true
}
```

### POST `/predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-Data-Consent: true" \
  -F "file=@ultrasound.png"
```
```json
{
  "device_used": "cpu",
  "threshold": 0.5,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "coverage_pct": 12.4,
  "mask_png_b64": "iVBORw0KGgo...",
  "overlay_png_b64": "iVBORw0KGgo..."
}
```

### POST `/explain/predict`
Returns segmentation mask + Grad-CAM heatmap showing which regions drove the prediction.
```json
{
  "mask_png_b64": "...",
  "overlay_png_b64": "...",
  "coverage_pct": 12.4,
  "xai": {
    "heatmap_b64": "...",
    "method": "gradcam",
    "clinical_note": "Red regions indicate areas that most strongly influenced the segmentation decision."
  },
  "request_id": "550e8400-..."
}
```

### GET `/gdpr/status`
Returns GDPR compliance status and data retention policy.

### DELETE `/gdpr/erase/{request_id}`
Permanently deletes all data associated with a request ID.
```json
{ "erased": true, "s3_deleted": true }
```

### GET `/metrics`
Prometheus metrics — request counts, latency histograms, error rates.

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Model + metrics only
pytest tests/test_suite.py -v

# API + middleware workflow
pytest tests/test_api.py -v
```

Test coverage:
- U-Net output shapes, gradient flow, no-NaN outputs
- Dice/IoU/precision/recall on known inputs
- GDPR consent enforcement (403 without header)
- Security headers on every response
- Full workflow: upload → predict → explain → erase

---

## 🧠 Training

### Dataset

[BUSI (Breast Ultrasound Images Dataset)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)  
780 images across 3 classes: benign, malignant, normal. Each image paired with a ground truth segmentation mask.

```
Data/
└── Dataset_BUSI_with_GT/
    ├── benign/
    ├── malignant/
    └── normal/
```

### Run Training

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best.pt
```

```
  Metric               Value    Threshold     Status
  ----------------------------------------------------
  dice                0.7300       0.7000    ✅ PASS
  iou                 0.6600       0.5500    ✅ PASS
  precision           0.8000       0.6500    ✅ PASS
  recall              0.7100       0.6500    ✅ PASS
```

---

## ⚙️ CI/CD Pipeline

```
Push to main/dev
      ↓
Code Quality (ruff + black)
      ↓
Unit + Integration Tests (pytest)
      ↓
Docker Build + Smoke Test (/health)
      ↓
Deploy to HuggingFace Spaces 🤗
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch |
| API Serving | FastAPI + Uvicorn |
| Production Middleware | medical-ai-middleware |
| Monitoring | Prometheus |
| GDPR & Compliance | GDPR middleware + AWS S3 |
| XAI | Grad-CAM |
| Experiment Tracking | W&B + MLflow |
| Data Versioning | DVC |
| Demo | Gradio + HuggingFace Spaces |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Dataset | BUSI (780 breast ultrasound images) |

---

## 📜 License

MIT License — free to use and modify.

---

## 🙋 Author

**Moez Bouassida** — AI/ML Engineer · Medical Imaging  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/moezbouassida/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/moebouassida)