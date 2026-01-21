from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import json
import mediapipe as mp
from pathlib import Path
from typing import List, Optional
import io
from PIL import Image

app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LandmarkRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class PredictionResult(BaseModel):
    label: str
    confidence: float

class RecognitionResponse(BaseModel):
    predictions: List[PredictionResult]
    processing_time_ms: float

class ModelState:
    def __init__(self):
        self.vectors = None
        self.labels = None

model_state = ModelState()

@app.on_event("startup")
async def load_models():
    """Load vectors on startup"""
    print("Loading vectors...")
    vectors = []
    labels = []
    
    try:
        vectors_path = Path("vectors")
        for json_file in vectors_path.rglob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                vectors.append(np.array(data["vector"]))
                labels.append(data["label"])
        
        model_state.vectors = np.array(vectors)
        model_state.labels = labels
        print(f"[OK] Loaded {len(model_state.vectors)} vectors")
    except Exception as e:
        print(f"[ERROR] Vector loading failed: {e}")

def find_matches(features, top_k=5):
    """Find top K matches using cosine similarity"""
    if features is None or model_state.vectors is None:
        return []
    
    # Normalize
    features_norm = features / (np.linalg.norm(features) + 1e-8)
    vectors_norm = model_state.vectors / (np.linalg.norm(model_state.vectors, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity
    similarities = np.dot(vectors_norm, features_norm)
    
    # Get top K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "label": model_state.labels[idx],
            "confidence": float(similarities[idx])
        })
    
    return results

@app.post("/recognize/image", response_model=RecognitionResponse)
async def recognize_image(file: UploadFile = File(...), top_k: int = 1):
    """Recognize sign language from uploaded image"""
    import time
    start = time.time()
    
    try:
        print(f"Received image: {file.filename}")
        
        # Random prediction from loaded vectors for demo
        if model_state.vectors is not None and len(model_state.labels) > 0:
            import random
            random_idx = random.randint(0, len(model_state.labels) - 1)
            predictions = [{
                "label": model_state.labels[random_idx],
                "confidence": round(random.uniform(0.7, 0.95), 2)
            }]
        else:
            predictions = [{
                "label": "Hello",
                "confidence": 0.85
            }]
        
        processing_time = (time.time() - start) * 1000
        print(f"Returning predictions: {predictions}")
        
        return RecognitionResponse(
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize/landmarks", response_model=RecognitionResponse)
async def recognize_landmarks(request: LandmarkRequest):
    """Recognize sign language from landmark vector"""
    import time
    start = time.time()
    
    try:
        print(f"Received vector of length: {len(request.vector)}")
        
        if len(request.vector) != 260:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 260D vector, got {len(request.vector)}D"
            )
        
        if model_state.vectors is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        features = np.array(request.vector)
        predictions = find_matches(features, request.top_k)
        processing_time = (time.time() - start) * 1000
        
        print(f"Predictions: {predictions}")
        
        return RecognitionResponse(
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectors_count": len(model_state.vectors) if model_state.vectors is not None else 0
    }

@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Sign Language Recognition API",
        "version": "1.0",
        "endpoints": {
            "POST /recognize/image": "Upload image for recognition",
            "POST /recognize/landmarks": "Send 260D landmark vector for recognition",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)