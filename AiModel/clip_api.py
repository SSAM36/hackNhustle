"""
CLIP IMAGE SEARCH API
=====================
FastAPI backend for image search using CLIP/SigLIP + Qdrant

Usage:
    uvicorn clip_api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /search - Upload image, get top matches
    GET /health - Health check
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
import numpy as np
import requests
import os
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModel
import io
from typing import List
import cv2
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

load_dotenv()

app = FastAPI(title="CLIP Image Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.qdrant_url = None
        self.headers = None
        self.collection = "hybrid_signs"  # Single collection with named vectors
        self.hands = None
        self.pose = None
        self.face = None
        
model_state = ModelState()

class SearchResult(BaseModel):
    label: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]
    processing_time_ms: float

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    print("Loading SigLIP model...")
    model_name = "google/siglip-large-patch16-384"  # 1024D - between base (768D) and SO400M (1152D)
    
    model_state.model = AutoModel.from_pretrained(model_name)
    model_state.processor = AutoProcessor.from_pretrained(model_name)
    model_state.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_state.model.to(model_state.device)
    model_state.model.eval()
    
    # MediaPipe Tasks API
    if MEDIAPIPE_AVAILABLE:
        try:
            hand_base = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
            hand_opts = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
            model_state.hands = vision.HandLandmarker.create_from_options(hand_opts)
            
            pose_base = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            pose_opts = vision.PoseLandmarkerOptions(base_options=pose_base)
            model_state.pose = vision.PoseLandmarker.create_from_options(pose_opts)
            
            face_base = python.BaseOptions(model_asset_path='models/face_landmarker.task')
            face_opts = vision.FaceLandmarkerOptions(base_options=face_base, num_faces=1)
            model_state.face = vision.FaceLandmarker.create_from_options(face_opts)
            
            print("✓ MediaPipe initialized (hands, pose, face)")
        except:
            model_state.hands = None
            model_state.pose = None
            model_state.face = None
            print("⚠ MediaPipe models not found")
    else:
        model_state.hands = None
        model_state.pose = None
        model_state.face = None
        
raw_url = os.getenv('q_url', 'http://localhost:6333').rstrip('/')

if 'cloud.qdrant.io' in raw_url:
    # Qdrant Cloud → HTTPS, no port
    if not raw_url.startswith('http'):
        raw_url = f"https://{raw_url}"
    model_state.qdrant_url = raw_url
else:
    # Local Qdrant
    model_state.qdrant_url = raw_url

api_key = os.getenv('q_api', '')
model_state.headers = {'Content-Type': 'application/json'}
if api_key:
    model_state.headers['api-key'] = api_key

print(f"✓ Model loaded on {model_state.device}")
print(f"✓ Qdrant: {model_state.qdrant_url}")

def crop_upper_body(image: Image.Image, padding: float = 0.2) -> Image.Image:
    """Crop to upper body with MediaPipe Tasks API"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    if not model_state.hands or not model_state.pose:
        crop_h = int(h * 0.6)
        start_y = (h - crop_h) // 2
        return Image.fromarray(img_array[start_y:start_y + crop_h, :])
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    hand_results = model_state.hands.detect(mp_img)
    pose_results = model_state.pose.detect(mp_img)
    
    all_x, all_y = [], []
    
    if hand_results.hand_landmarks:
        for hand_landmarks in hand_results.hand_landmarks:
            for lm in hand_landmarks:
                all_x.append(int(lm.x * w))
                all_y.append(int(lm.y * h))
    
    if pose_results.pose_landmarks:
        for landmarks in pose_results.pose_landmarks:
            for idx in [11, 12, 13, 14]:
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    all_x.append(int(lm.x * w))
                    all_y.append(int(lm.y * h))
    
    if not all_x:
        return image
    
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)
    
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    
    return Image.fromarray(img_array[y_min:y_max, x_min:x_max])

def extract_landmarks(image: Image.Image) -> np.ndarray:
    """Extract 358D landmarks (hands + pose + face + geometric)"""
    if not model_state.hands or not model_state.pose:
        return np.zeros(358)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img_array = cv2.resize(img_array, (int(w*scale), int(h*scale)))
    
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    hand_results = model_state.hands.detect(mp_img)
    pose_results = model_state.pose.detect(mp_img)
    face_results = model_state.face.detect(mp_img) if model_state.face else None
    
    landmarks = np.zeros(138)
    geometric = []
    
    # Hands (126D)
    if hand_results.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.hand_landmarks[:2]):
            offset = i * 63
            wrist = hand_landmarks[0]
            for j, lm in enumerate(hand_landmarks):
                landmarks[offset + j*3] = lm.x - wrist.x
                landmarks[offset + j*3 + 1] = lm.y - wrist.y
                landmarks[offset + j*3 + 2] = lm.z - wrist.z
            
            tips = [hand_landmarks[i] for i in [4, 8, 12, 16, 20]]
            for i in range(len(tips)):
                for j in range(i+1, len(tips)):
                    geometric.append(np.sqrt((tips[i].x-tips[j].x)**2+(tips[i].y-tips[j].y)**2+(tips[i].z-tips[j].z)**2))
    
    # Pose (12D)
    if pose_results.pose_landmarks:
        for landmarks_list in pose_results.pose_landmarks:
            left_elbow = landmarks_list[13]
            right_elbow = landmarks_list[14]
            center_x = (left_elbow.x + right_elbow.x) / 2
            center_y = (left_elbow.y + right_elbow.y) / 2
            center_z = (left_elbow.z + right_elbow.z) / 2
            for i, idx in enumerate([11, 12, 13, 14]):
                lm = landmarks_list[idx]
                landmarks[126 + i*3] = lm.x - center_x
                landmarks[126 + i*3 + 1] = lm.y - center_y
                landmarks[126 + i*3 + 2] = lm.z - center_z
    
    # Face (60D)
    face_features = np.zeros(60)
    if face_results and face_results.face_landmarks:
        face_landmarks = face_results.face_landmarks[0]
        nose = face_landmarks[1]
        key_indices = [1, 33, 61, 291, 199, 10, 152, 234, 454, 93, 323, 168, 6, 197, 195, 5, 4, 263, 362, 386]
        for i, idx in enumerate(key_indices):
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                face_features[i*3] = lm.x - nose.x
                face_features[i*3 + 1] = lm.y - nose.y
                face_features[i*3 + 2] = lm.z - nose.z
    
    while len(geometric) < 160:
        geometric.append(0.0)
    
    return np.concatenate([landmarks, geometric[:160], face_features])

def encode_image(image: Image.Image) -> np.ndarray:
    """Encode image to vector with enhanced preprocessing"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy for preprocessing
    img_array = np.array(image)
    
    # Apply histogram equalization for better contrast
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Slight denoising while preserving edges
    img_array = cv2.bilateralFilter(img_array, 5, 50, 50)
    
    # Convert back to PIL and resize
    image = Image.fromarray(img_array)
    image = image.resize((384, 384), Image.Resampling.LANCZOS)
    
    inputs = model_state.processor(images=image, return_tensors="pt").to(model_state.device)
    
    with torch.no_grad():
        features = model_state.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy().flatten()

def search_qdrant(img_vector: np.ndarray, landmark_vector: np.ndarray, top_k: int = 15, alpha: float = 0.6, beta: float = 0.4) -> List[dict]:
    """Search using hybrid named vectors"""
    try:
        r1 = requests.post(
            f"{model_state.qdrant_url}/collections/{model_state.collection}/points/search",
            json={"vector": {"name": "clip", "vector": img_vector.tolist()}, "limit": top_k * 2, "with_payload": True},
            headers=model_state.headers,
            timeout=5
        )
        
        r2 = requests.post(
            f"{model_state.qdrant_url}/collections/{model_state.collection}/points/search",
            json={"vector": {"name": "landmarks", "vector": landmark_vector.tolist()}, "limit": top_k * 2, "with_payload": True},
            headers=model_state.headers,
            timeout=5
        )
        
        combined = {}
        if r1.status_code == 200:
            for hit in r1.json().get('result', []):
                label = hit["payload"]["label"]
                combined[label] = alpha * hit["score"]
        
        if r2.status_code == 200:
            for hit in r2.json().get('result', []):
                label = hit["payload"]["label"]
                combined[label] = combined.get(label, 0) + beta * hit["score"]
        
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"label": label, "score": min(score, 1.0)} for label, score in sorted_results]
    except Exception as e:
        print(f"Search error: {e}")
        return []

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model_loaded": model_state.model is not None,
        "device": model_state.device,
        "collection": model_state.collection
    }

@app.post("/search", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...), top_k: int = 8, alpha: float = 0.6, beta: float = 0.4):
    """
    Upload an image and get top matching labels
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        top_k: Number of results to return (default: 5)
    
    Returns:
        SearchResponse with results and processing time
    """
    import time
    start = time.time()
    
    if not model_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Crop to upper body
        cropped = crop_upper_body(image, padding=0.2)
        
        # Encode image and landmarks
        img_vector = encode_image(cropped)
        landmark_vector = extract_landmarks(image)
        
        # Search
        results = search_qdrant(img_vector, landmark_vector, top_k, alpha, beta)
        
        processing_time = (time.time() - start) * 1000
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/search/batch", response_model=List[SearchResponse])
async def search_batch(files: List[UploadFile] = File(...), top_k: int = 5):
    """
    Upload multiple images and get results for each
    
    Args:
        files: List of image files
        top_k: Number of results per image
    
    Returns:
        List of SearchResponse objects
    """
    import time
    
    if not model_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    
    for file in files:
        start = time.time()
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            cropped = crop_upper_body(image, padding=0.2)
            img_vector = encode_image(cropped)
            landmark_vector = extract_landmarks(image)
            results = search_qdrant(img_vector, landmark_vector, top_k, 0.6, 0.4)
            processing_time = (time.time() - start) * 1000
            
            responses.append(SearchResponse(
                results=[SearchResult(**r) for r in results],
                processing_time_ms=round(processing_time, 2)
            ))
        except Exception as e:
            responses.append(SearchResponse(
                results=[],
                processing_time_ms=0
            ))
    
    return responses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)