from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import ModelWrapper
from PIL import Image
import io
import torchvision.transforms as T

app = FastAPI(title="Breast Ultrasound Segmentation API")

MODEL_NAME = "BreastSeg"   # This is the MLflow registered model name
model_wrapper = ModelWrapper(MODEL_NAME, device="cpu")

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_tensor = transform(image).unsqueeze(0)
    pred_mask = model_wrapper.predict(image_tensor)

    # Return mask as a list for demonstration (can be changed to encoded image etc.)
    mask_list = pred_mask.squeeze().cpu().numpy().tolist()
    return {"prediction_mask": mask_list}
