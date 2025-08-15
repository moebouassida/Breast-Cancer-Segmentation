# 🩺 Breast Ultrasound Segmentation

Semantic segmentation of breast ultrasound images — from a humble Jupyter notebook in TensorFlow to a production-ready PyTorch service with experiment tracking, model registry, API endpoints, Docker deployment, and an interactive Gradio demo.

---

## 📖 Project Story
This project started as a student experiment in a Jupyter notebook.  
At the time, there was:
- no API
- no deployment
- no experiment tracking
- everything was run manually in TensorFlow

Now, it has grown into a full ML engineering project with:
- PyTorch UNet model for segmentation
- MLflow for experiment tracking & model registry
- FastAPI REST API for inference
- Gradio web demo for quick testing
- Docker for containerized deployment
- GPU support via NVIDIA Container Toolkit
- Clean project structure with reusable components

---

## 📂 Project Structure

.
├── app.py                 # FastAPI + Gradio app
├── Dockerfile             # CPU deployment image
├── Dockerfile.gpu         # GPU deployment image
├── requirements.txt       # Python dependencies (CPU)
├── requirements-gpu.txt   # Python dependencies (GPU)
├── src/
│   ├── config.py          # Configurations
│   ├── model.py           # UNet implementation
│   ├── validate.py        # Validation loop
│   ├── mlflow_utils.py    # MLflow logging helpers
│   ├── utils.py           # Visualization helpers
│   ├── metrics.py         # Segmentation metrics
├── Data/
│   ├── data_loader.py     # Dataset & dataloader
├── Experiments/mlruns/    # MLflow local tracking store
└── README.md              # This file

---

## 🚀 Features

- PyTorch UNet architecture for binary segmentation.
- Training loop with:
  - BCEWithLogitsLoss
  - Dice, IoU, Pixel Accuracy, Precision, Recall metrics
  - MLflow metric logging & artifact storage
  - Automatic best model checkpointing
- Validation with visual overlays & threshold sweep plots.
- FastAPI REST API:
  - `/health` — check model/device
  - `/predict` — upload image → get mask & overlay
- Gradio UI at `/gradio` for in-browser testing.
- Docker for reproducible deployments (CPU/GPU).
- MLflow Model Registry integration for loading production models.

---

## ⚙️ Setup & Installation

### 1. Clone the repo
git clone https://github.com/<your-username>/breast-ultrasound-segmentation.git
cd breast-ultrasound-segmentation

### 2. Install dependencies (CPU)
pip install --no-cache-dir -r requirements.txt

For GPU:
pip install --no-cache-dir -r requirements-gpu.txt

---

## 🏋️‍♀️ Training

1. Configure training parameters in `src/config.py` (batch size, learning rate, epochs, etc.).
2. Run training:
python train.py

3. MLflow logs will be stored locally in:
Experiments/mlruns

---

## 🌐 Running the API & Gradio UI

### CPU (local)
uvicorn app:app --host 0.0.0.0 --port 8000

- API docs: http://localhost:8000/docs  
- Gradio UI: http://localhost:8000/gradio

---

## 🐳 Docker Deployment

### Build (CPU)
docker build -t breast-seg-api .

### Run (CPU)
docker run --rm -p 8000:8000 \
  -e MODEL_NAME=BreastSeg \
  -e MLFLOW_TRACKING_URI=file:/app/Experiments/mlruns \
  -v $(pwd)/Experiments/mlruns:/app/Experiments/mlruns:ro \
  breast-seg-api

### Build (GPU)
docker build -f Dockerfile.gpu -t breast-seg-api:gpu .

### Run (GPU)
docker run --rm -it --gpus all -p 8000:8000 \
  -e MODEL_NAME=BreastSeg \
  -e MLFLOW_TRACKING_URI=file:/app/Experiments/mlruns \
  -v $(pwd)/Experiments/mlruns:/app/Experiments/mlruns:ro \
  breast-seg-api:gpu

---

## 📊 Using MLflow Model Registry

1. During training, models are logged to MLflow.
2. Promote the best model to `Production` in the MLflow UI.
3. The API automatically loads:
models:/BreastSeg/Production

---

## 💻 API Usage Example

curl -X POST "http://localhost:8000/predict?return_images=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png"

---

## 🖥️ Gradio Demo
Go to:
http://localhost:8000/gradio

Upload a grayscale ultrasound image, view:
- Predicted binary mask
- Overlay on original image

---

## 🛠️ Technologies

- PyTorch — Deep learning framework
- MLflow — Experiment tracking & model registry
- FastAPI — REST API framework
- Gradio — Interactive UI for demos
- Docker — Containerized deployment
- NVIDIA Container Toolkit — GPU passthrough in Docker

---

## 📜 License
MIT License — feel free to use and modify for your own projects.
