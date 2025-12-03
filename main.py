from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
import librosa
from librosa.feature import melspectrogram
from io import BytesIO
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import uvicorn
import os
import sys
import threading

from model import genreNet 
from config import GENRES 

app = FastAPI(title="Audio Genre Classification API")

MODEL_PATH = "net.pt"
DEVICE = "cpu"

# Global variables
le = None
model = None
_loading = False
_load_lock = threading.Lock()

def ensure_model_loaded():
    """Load model if not already loaded - thread-safe"""
    global model, le, _loading
    
    if model is not None and le is not None:
        return True
    
    with _load_lock:
        # Double-check after acquiring lock
        if model is not None and le is not None:
            return True
            
        if _loading:
            return False
        
        _loading = True
        
        try:
            print("=" * 50, flush=True)
            print("LOADING: Starting model load", flush=True)
            print(f"LOADING: Python {sys.version[:20]}...", flush=True)
            print(f"LOADING: Directory: {os.getcwd()}", flush=True)
            
            # Label Encoder
            print("LOADING: Label Encoder...", flush=True)
            le = LabelEncoder().fit(GENRES)
            print(f"LOADING: ✓ Encoder ready", flush=True)
            
            # Model
            print("LOADING: Model architecture...", flush=True)
            model = genreNet()
            print("LOADING: ✓ Architecture ready", flush=True)
            
            # Weights
            if os.path.exists(MODEL_PATH):
                print(f"LOADING: Weights from {MODEL_PATH}...", flush=True)
                state_dict = torch.load(MODEL_PATH, map_location=torch.device(DEVICE), weights_only=False)
                model.load_state_dict(state_dict)
                model.eval()
                print("LOADING: ✓ Weights loaded", flush=True)
            else:
                print("LOADING: ⚠ Model file not found", flush=True)
                model = None
                return False
                
            print("=" * 50, flush=True)
            print("LOADING COMPLETE", flush=True)
            print("=" * 50, flush=True)
            return True
            
        except Exception as e:
            print(f"LOADING ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            model = None
            le = None
            return False
        finally:
            _loading = False

# Helper Functions
def preprocess_audio(audio_bytes: bytes, sr=22050):
    try:
        y, sr = librosa.load(BytesIO(audio_bytes), mono=True, sr=sr)
        S = melspectrogram(y=y, sr=sr).T
        S = S[:-1 * (S.shape[0] % 128)]
        
        if S.shape[0] == 0:
            return None

        num_chunk = int(S.shape[0] / 128)
        data_chunks = np.split(S, num_chunk)
        return data_chunks
    except Exception as e:
        print(f"PREPROCESS ERROR: {e}", flush=True)
        return None

def predict_genre(data_chunks):
    genres = []
    if model is None or le is None:
        return []
        
    with torch.no_grad():
        for data in data_chunks:
            data_tensor = torch.FloatTensor(data).view(1, 1, 128, 128).to(DEVICE)
            preds = model(data_tensor)
            pred_val, pred_index = preds.max(1)
            pred_index = pred_index.cpu().numpy()
            pred_val = np.exp(pred_val.cpu().numpy()[0])
            
            pred_genre = le.inverse_transform(pred_index).item()
            
            if pred_val >= 0.5:
                genres.append(pred_genre)
    return genres

# API Endpoints
@app.get("/")
def home():
    # Try to load model on first request
    ensure_model_loaded()
    
    return {
        "message": "Genre Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "encoder_loaded": le is not None,
        "endpoints": ["/health", "/predict"]
    }

@app.get("/health")
def health():
    # Try to load model
    ensure_model_loaded()
    
    return {
        "status": "healthy",
        "model": model is not None,
        "encoder": le is not None,
        "loading": _loading
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ensure model is loaded
    if not ensure_model_loaded():
        raise HTTPException(503, "Model is loading, please try again")
    
    if model is None or le is None:
        raise HTTPException(503, "Service not ready")
    
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Must be audio file")

    try:
        audio_bytes = await file.read()
        chunks = preprocess_audio(audio_bytes)
        
        if not chunks:
            return JSONResponse({"error": "Audio too short"}, 400)

        predicted_genres = predict_genre(chunks)
        
        if not predicted_genres:
            return {"result": "No genre detected"}

        count = Counter(predicted_genres)
        total = sum(count.values())
        
        results = []
        for genre, freq in count.most_common():
            percentage = (freq / total) * 100
            results.append({"genre": genre, "confidence": f"{percentage:.2f}%"})

        return {
            "filename": file.filename,
            "top_prediction": results[0]['genre'],
            "breakdown": results
        }

    except Exception as e:
        print(f"PREDICT ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, 500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)