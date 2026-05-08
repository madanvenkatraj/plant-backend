from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import os
import json
import time
from PIL import Image
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from gtts import gTTS
import base64

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize New Gen AI Client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Plant AI Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://plant-project-330a0.web.app",
        "https://plant-project-330a0.firebaseapp.com",
        "http://localhost:5173",
        "http://localhost:3000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
class_indices = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "plant_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "class_indices.json")

@app.on_event("startup")
async def load_model_on_startup():
    global model, class_indices
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INDICES_PATH):
            import tensorflow as tf
            
            class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
                def __init__(self, **kwargs):
                    for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                        kwargs.pop(key, None)
                    super().__init__(**kwargs)

            model = tf.keras.models.load_model(
                MODEL_PATH, 
                custom_objects={'BatchNormalization': PatchedBatchNormalization}
            )
            with open(CLASS_INDICES_PATH, "r") as f:
                class_indices = json.load(f)
            class_indices = {str(v): k for k, v in class_indices.items()}
            print(f"[OK] Model loaded. Classes: {len(class_indices)}")
        else:
            print("[WARN] Model files not found.")
    except Exception as e:
        print(f"[ERROR] Model load error: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
async def root():
    return {"message": "Plant AI Backend v1.1 is running!", "model_loaded": model is not None, "ai_client": client is not None}

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if model is None or class_indices is None:
        import random
        mock_classes = ["Apple___Apple_scab", "Tomato___healthy", "Potato___Early_blight"]
        label = random.choice(mock_classes)
        return JSONResponse({"label": label, "confidence": 95.50, "mock": True})

    try:
        img_array = preprocess_image(contents)
        predictions = model.predict(img_array, verbose=0)
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        label = class_indices.get(str(pred_idx), "Unknown")

        return JSONResponse({
            "label": label,
            "predicted_class": label,
            "confidence": round(confidence, 2),
            "class_index": pred_idx
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ─── GEMINI AI ENDPOINTS (REFACTORED FOR GOOGLE-GENAI SDK) ────────

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

class TranslateRequest(BaseModel):
    data: dict
    target_language: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini AI Client not initialized.")
    
    # Priority order for 2026 deployment
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    last_error = ""

    for model_name in models_to_try:
        try:
            prompt = f"""You are a highly knowledgeable agricultural specialist. 
Reply in {request.language}. Question: {request.message}"""
            
            response = client.models.generate_content(model=model_name, contents=prompt)
            return {"response": response.text, "model_used": model_name}
        except Exception as e:
            last_error = str(e)
            print(f"[CHAT DEBUG] Model {model_name} failed: {e}")
            time.sleep(1) # Small delay before retry
            continue

    return {"response": f"AI Brain Connection Error. All models failed. Error: {last_error}"}

@app.post("/translate")
async def translate(request: TranslateRequest):
    if not client:
        return {"translated_data": request.data}
    
    # Using the most robust Flash models for high-speed translation
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
    
    for model_name in models_to_try:
        try:
            # More explicit prompt to ensure valid JSON structure and FULL content translation
            prompt = (
                f"You are a professional agricultural translator. "
                f"Translate all the values in the following JSON data into {request.target_language}. "
                f"Crucially, translate the detailed descriptions for causes, symptoms, treatment, and prevention. "
                "Keep the JSON keys (plantName, diseaseName, causes, symptoms, treatment, prevention) EXACTLY the same. "
                "Return ONLY the translated JSON object, nothing else.\n\n"
                f"Data to translate: {json.dumps(request.data)}"
            )
            response = client.models.generate_content(model=model_name, contents=prompt)
            
            text = response.text.replace("```json", "").replace("```", "").strip()
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                translated_json = json.loads(text[start:end+1])
                # Ensure all original keys are present in the response
                for key in request.data.keys():
                    if key not in translated_json:
                        translated_json[key] = request.data[key]
                return {"translated_data": translated_json, "model_used": model_name}
        except Exception as e:
            print(f"[TRANSLATE DEBUG] {model_name} failed: {e}")
            continue
    return {"translated_data": request.data}

class TTSRequest(BaseModel):
    text: str
    lang: str

@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        clean_text = request.text.replace("*", "").replace("#", "").strip()
        lang_map = {"en": "en", "ta": "ta", "hi": "hi", "te": "te", "kn": "kn"}
        tts_obj = gTTS(text=clean_text, lang=lang_map.get(request.lang, "en"), slow=False)
        audio_buffer = io.BytesIO()
        tts_obj.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return {"audioContent": base64.b64encode(audio_buffer.read()).decode("utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
