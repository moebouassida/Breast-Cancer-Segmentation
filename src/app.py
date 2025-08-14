from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import ModelWrapper
from PIL import Image
import io
import torchvision.transforms as T
import torch

app = FastAPI(title="Breast Ultrasound Segmentation API")

# Automatically select GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BreastSeg"   # MLflow registered model name

model_wrapper = ModelWrapper(MODEL_NAME, device=DEVICE)

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

@app.get("/health")
def health_check():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = model_wrapper.predict(image_tensor)

    # Convert mask to list (you could also return an image file)
    mask_list = pred_mask.squeeze().cpu().numpy().tolist()
    return {
        "prediction_mask": mask_list,
        "device_used": DEVICE
    }
