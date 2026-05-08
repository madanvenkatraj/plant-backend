from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import os
import json
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from gtts import gTTS
import base64

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="Plant AI Backend", version="1.0.0")

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
async def load_model():
    global model, class_indices
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INDICES_PATH):
            import tensorflow as tf
            
            # Patch for BatchNormalization compatibility between Keras versions
            class PatchedBatchNormalization(tf.keras.layers.BatchNormalization):
                def __init__(self, **kwargs):
                    # Remove arguments that cause "Unrecognized keyword arguments" in newer Keras/TF
                    for key in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                        kwargs.pop(key, None)
                    super().__init__(**kwargs)

            model = tf.keras.models.load_model(
                MODEL_PATH, 
                custom_objects={'BatchNormalization': PatchedBatchNormalization}
            )
            with open(CLASS_INDICES_PATH, "r") as f:
                class_indices = json.load(f)
            # Invert: {class_name: index} -> {index: class_name}
            class_indices = {str(v): k for k, v in class_indices.items()}
            print(f"[OK] Model loaded. Classes: {len(class_indices)}")
        else:
            print("[WARN] Model files not found. Run 'python train.py' first.")
    except Exception as e:
        print(f"[ERROR] Model load error: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
async def root():
    return {"message": "Plant AI Backend is running!", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    if model is None or class_indices is None:
        # Return a mock prediction if the model is not trained yet
        import random
        mock_classes = ["Apple___Apple_scab", "Tomato___healthy", "Potato___Early_blight", "Corn_(maize)___Common_rust_"]
        label = random.choice(mock_classes)
        return JSONResponse({
            "label": label,
            "predicted_class": label,
            "confidence": 95.50,
            "confidence_score": 95.50,
            "class_index": 0,
            "mock": True
        })

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
            "confidence_score": round(confidence, 2),
            "class_index": pred_idx
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ─── GEMINI AI ENDPOINTS ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    language: str = "English"

class TranslateRequest(BaseModel):
    data: dict
    target_language: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")
    
    # Flash models have the highest free-tier quotas and are most reliable for chatbots
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash"]
    last_error = ""

    for model_name in models_to_try:
        try:
            model_ai = genai.GenerativeModel(model_name)
            prompt = f"""You are a highly knowledgeable agricultural expert and plant disease specialist AI assistant.
Your goal is to answer questions strictly related to plants, agriculture, farming, crops, trees, and plant diseases.
The user is currently communicating in language: {request.language}. Please reply in this language.
If the question is completely unrelated to plants, agriculture, or nature, politely refuse to answer.
Make your answers concise, well-structured, using bullet points and emojis where appropriate.

User question: {request.message}
"""
            response = model_ai.generate_content(prompt)
            return {"response": response.text, "model_used": model_name}
        except Exception as e:
            last_error = str(e)
            print(f"[CHAT DEBUG] Model {model_name} failed: {e}")
            continue

    # If all models fail
    error_msg = f"[CHAT ERROR] All models failed. Last error: {last_error}"
    print(error_msg)
    with open("backend_error.log", "a") as f:
        f.write(error_msg + "\n")
    return {"response": f"I'm sorry, I'm having trouble connecting to my AI brain. (All models failed). Error: {last_error}"}

@app.post("/translate")
async def translate(request: TranslateRequest):
    if not GEMINI_API_KEY:
        return {"translated_data": request.data}
    
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash"]
    
    for model_name in models_to_try:
        try:
            model_ai = genai.GenerativeModel(model_name)
            # Be extremely specific about keys and language
            prompt = f"Translate this JSON to {request.target_language}. Return ONLY valid JSON with keys: plantName, diseaseName, causes, symptoms, treatment, prevention.\n\nData: {json.dumps(request.data)}"
            
            response = model_ai.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            
            # Find the first { and last } to handle stray text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]
                
            translated_json = json.loads(text)
            return {"translated_data": translated_json, "model_used": model_name}
        except Exception as e:
            print(f"[TRANSLATE DEBUG] Model {model_name} failed: {e}")
            continue
    return {"translated_data": request.data}

class TTSRequest(BaseModel):
    text: str
    lang: str

@app.post("/tts")
async def tts(request: TTSRequest):
    try:
        # Clean text of markdown
        clean_text = request.text.replace("*", "").replace("#", "").replace("_", "").strip()
        
        # Mapping frontend lang codes to gTTS lang codes
        lang_map = {
            "en": "en",
            "ta": "ta",
            "hi": "hi",
            "te": "te",
            "kn": "kn"
        }
        tts_lang = lang_map.get(request.lang, "en")
        
        tts_obj = gTTS(text=clean_text, lang=tts_lang, slow=False)
        
        # Save to a temporary buffer
        audio_buffer = io.BytesIO()
        tts_obj.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        
        return {"audioContent": audio_base64}
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
