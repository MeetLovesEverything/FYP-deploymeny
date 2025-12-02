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

from model import genreNet 
from config import GENRES 

app = FastAPI(title="Audio Genre Classification API")

MODEL_PATH = "net.pt"
DEVICE = "cpu"

# Initialize Label Encoder
le = LabelEncoder().fit(GENRES)

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = genreNet()
    try:
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
            model.eval()
            print(f"-> Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"WARNING: {MODEL_PATH} not found. Please upload your model file.")
    except Exception as e:
        print(f"Error loading model: {e}")

# --- Helper Functions ---
def preprocess_audio(audio_bytes: bytes, sr=22050):
    try:
        # Load audio from bytes
        y, sr = librosa.load(BytesIO(audio_bytes), mono=True, sr=sr)
        
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
        print(f"Error in preprocessing: {e}")
        return None

def predict_genre(data_chunks):
    genres = []
    if model is None:
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

@app.get("/")
def home():
    return {"message": "Genre Classification API is running. Send POST request to /predict"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file.")

    try:
        audio_bytes = await file.read()
        chunks = preprocess_audio(audio_bytes)
        
        if not chunks:
            return JSONResponse(content={"error": "Audio processing failed or audio too short"}, status_code=400)

        predicted_genres = predict_genre(chunks)
        
        if not predicted_genres:
            return {"result": "No genre detected with high confidence."}

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
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)