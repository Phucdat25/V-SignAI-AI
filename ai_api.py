import os
import uuid
import shutil
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from sign_service import VSignExtractorAndPredictor


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ctrgcn_model.onnx"
GLOSS_PATH = BASE_DIR / "gloss_to_id.json"
TEMP_DIR = BASE_DIR / "temp_uploads"

TEMP_DIR.mkdir(exist_ok=True)


app = FastAPI(
    title="VSign AI Service",
    description="AI service for sign language video recognition",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


service = None


@app.on_event("startup")
def startup_event():
    global service

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not GLOSS_PATH.exists():
        raise RuntimeError(f"Gloss file not found: {GLOSS_PATH}")

    service = VSignExtractorAndPredictor(
        onnx_model_path=str(MODEL_PATH),
        gloss_to_id_path=str(GLOSS_PATH)
    )

    print("AI model loaded successfully.")


@app.on_event("shutdown")
def shutdown_event():
    global service

    if service is not None:
        service.close()
        print("AI service closed.")


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "VSign AI Service is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": service is not None
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    if service is None:
        raise HTTPException(status_code=500, detail="AI model is not loaded")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_extensions}"
        )

    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = TEMP_DIR / temp_filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = service.predict_isolated_word(
            video_path=str(temp_path)
        )

        return result

    except Exception as e:
        print("========== AI ERROR ==========")
        traceback.print_exc()
        print("==============================")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
