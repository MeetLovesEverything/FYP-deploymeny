from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import numpy as np
import librosa
from librosa.feature import melspectrogram
from io import BytesIO
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import os

from model import genreNet 
from config import GENRES 

print("=" * 60, flush=True)
print("APP INIT: Starting application initialization", flush=True)

app = FastAPI(title="Music Genre Classification API")

# Configuration
MODEL_PATH = "net.pt"
DEVICE = "cpu"

# Initialize label encoder at module level (lightweight)
le = LabelEncoder().fit(GENRES)
print(f"APP INIT: Label encoder ready with {len(GENRES)} genres", flush=True)

# Model will be loaded on first request (lazy loading)
model = None

print("APP INIT: FastAPI app created successfully", flush=True)
print("=" * 60, flush=True)

def load_model_if_needed():
    """Load model on first request - lazy loading"""
    global model
    
    if model is not None:
        return True
    
    try:
        print("MODEL: Loading model for first time...", flush=True)
        model_instance = genreNet()
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=torch.device(DEVICE), weights_only=False)
            model_instance.load_state_dict(state_dict)
            model_instance.eval()
            model = model_instance
            print("MODEL: ✓ Model loaded successfully", flush=True)
            return True
        else:
            print(f"MODEL: ERROR - {MODEL_PATH} not found", flush=True)
            return False
            
    except Exception as e:
        print(f"MODEL: ERROR loading model - {e}", flush=True)
        return False

def extract_features(audio_bytes: bytes):
    """Extract mel spectrogram features from audio bytes"""
    try:
        # Load audio
        y, sr = librosa.load(BytesIO(audio_bytes), mono=True, sr=22050)
        
        # Get Mel Spectrogram
        S = melspectrogram(y=y, sr=sr).T
        
        # Trim to fit 128 width chunks
        S = S[:-1 * (S.shape[0] % 128)]
        
        if S.shape[0] == 0:
            return None

        # Split into chunks
        num_chunk = int(S.shape[0] / 128)
        data_chunks = np.split(S, num_chunk)
        return data_chunks
        
    except Exception as e:
        print(f"FEATURE EXTRACTION ERROR: {e}", flush=True)
        return None

def predict_genres(data_chunks):
    """Predict genres from audio chunks"""
    if model is None:
        return []
    
    genres = []
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

@app.get("/")
def home():
    return {
        "message": "Music Genre Classification API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict-genre"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict-genre")
async def predict_genre(audio_file: UploadFile = File(...)):
    """Predict music genre from audio file"""
    try:
        # Load model if not already loaded
        if not load_model_if_needed():
            return JSONResponse(
                content={"error": "Model not available"},
                status_code=503
            )
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Extract features
        chunks = extract_features(audio_bytes)
        if not chunks:
            return JSONResponse(
                content={"error": "Audio processing failed or audio too short"},
                status_code=400
            )
        
        # Predict genres
        predicted_genres = predict_genres(chunks)
        
        if not predicted_genres:
            return JSONResponse(
                content={"result": "No genre detected with high confidence"},
                status_code=200
            )
        
        # Calculate percentages
        count = Counter(predicted_genres)
        total = sum(count.values())
        
        results = []
        for genre, freq in count.most_common():
            percentage = (freq / total) * 100
            results.append({
                "genre": genre,
                "confidence": f"{percentage:.2f}%"
            })
        
        return JSONResponse(content={
            "filename": audio_file.filename,
            "predicted_genre": results[0]['genre'],
            "all_predictions": results
        })
        
    except Exception as e:
        print(f"PREDICTION ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

print("APP INIT: All routes registered", flush=True)
print("APP INIT: Application ready to start", flush=True)