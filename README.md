# 🩺 Breast Ultrasound Segmentation

Semantic segmentation of breast ultrasound images — trained with PyTorch and deployed as a production-ready service with FastAPI, Docker, and an interactive Gradio demo.

---

## 📸 Demo Screenshot
![Gradio Demo](assets/demo.png)

---

## 📖 Project Story
This project started as a simple experiment in a Jupyter notebook.  
At the time there was:
- no API  
- no deployment  
- everything was run manually  

Now, it has grown into a full ML engineering project with:
- PyTorch UNet model for segmentation  
- FastAPI REST API for inference  
- Gradio web demo for quick testing  
- Docker for containerized deployment  
- GPU support via NVIDIA Container Toolkit  
- Clean project structure with reusable components  

---

## 📂 Project Structure

.
├── Docker/
│   └── Dockerfile       # GPU-ready deployment image
├── requirements.txt     # Python dependencies
├── src/
│   ├── app.py           # FastAPI + Gradio app
│   ├── model.py         # UNet implementation
│   ├── utils.py         # Pre/post-processing
├── checkpoints/
│   └── best.pt          # Trained model weights
└── README.md            # This file

---

## 🚀 Features

- PyTorch UNet architecture for binary segmentation.  
- FastAPI REST API:
  - `/health` — check model/device  
  - `/predict` — upload image → get mask & overlay  
- Gradio UI at `/gradio` for in-browser testing.  
- Docker for reproducible deployments (CPU/GPU).  

---

## ⚙️ Setup & Installation

### 1. Clone the repo
git clone https://github.com/<your-username>/breast-ultrasound-segmentation.git
cd breast-ultrasound-segmentation

### 2. Install dependencies (local)
pip install --no-cache-dir -r requirements.txt

---

## 🌐 Running the API & Gradio UI

### Local (with Python)
uvicorn src.app:app --host 0.0.0.0 --port 8000

- Gradio UI: http://localhost:8000/gradio

### With Docker (CPU)
docker build -f Docker/Dockerfile -t breast-seg-api .
docker run --rm -p 8000:8000 breast-seg-api

### With Docker (GPU)
docker run --rm -it --gpus all -p 8000:8000 breast-seg-api

---

## 💻 API Usage Example

curl -X POST "http://localhost:8000/predict?return_images=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.png"

---

## 🛠️ Technologies

- PyTorch — Deep learning framework  
- FastAPI — REST API framework  
- Gradio — Interactive UI for demos  
- Docker — Containerized deployment  
- NVIDIA Container Toolkit — GPU passthrough in Docker  

---

## 📜 License
MIT License — feel free to use and modify for your own projects.
