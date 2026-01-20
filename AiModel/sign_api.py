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
        self.hands = None
        self.pose = None
        self.face = None
        self.vectors = None
        self.labels = None

model_state = ModelState()

@app.on_event("startup")
async def load_models():
    """Load MediaPipe models and vectors on startup"""
    print("Loading MediaPipe models...")
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    model_state.hands = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
    )
    model_state.pose = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5
        )
    )
    model_state.face = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
    )
    
    print("Loading vectors...")
    vectors = []
    labels = []
    
    vectors_path = Path("vectors")
    for json_file in vectors_path.rglob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            vectors.append(np.array(data["vector"]))
            labels.append(data["label"])
    
    model_state.vectors = np.array(vectors)
    model_state.labels = labels
    
    print(f"✓ Loaded {len(model_state.vectors)} vectors")

def extract_features(frame):
    """Extract normalized features from frame"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    hands_result = model_state.hands.detect(mp_image)
    pose_result = model_state.pose.detect(mp_image)
    face_result = model_state.face.detect(mp_image)
    
    # Extract face center
    face_center = None
    if face_result.face_landmarks:
        nose = face_result.face_landmarks[0][1]
        face_center = np.array([nose.x, nose.y, nose.z])
    
    if face_center is None:
        return None
    
    features = []
    
    # Hands
    if hands_result.hand_landmarks:
        for hand_landmarks in hands_result.hand_landmarks:
            wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
            
            for i in range(21):
                lm = hand_landmarks[i]
                point = np.array([lm.x, lm.y, lm.z])
                wrist_dist = np.linalg.norm(point - wrist)
                face_dist = np.linalg.norm(point - face_center)
                rel_x = point[0] - wrist[0]
                rel_y = point[1] - wrist[1]
                rel_z = point[2] - wrist[2]
                features.extend([rel_x, rel_y, rel_z, wrist_dist, face_dist])
    else:
        features.extend([0.0] * (21 * 5 * 2))
    
    if hands_result.hand_landmarks and len(hands_result.hand_landmarks) == 1:
        features.extend([0.0] * (21 * 5))
    
    # Pose
    if pose_result.pose_landmarks:
        pose_indices = [11, 12, 13, 14, 15, 16]
        left_shoulder = np.array([
            pose_result.pose_landmarks[0][11].x,
            pose_result.pose_landmarks[0][11].y,
            pose_result.pose_landmarks[0][11].z
        ])
        right_shoulder = np.array([
            pose_result.pose_landmarks[0][12].x,
            pose_result.pose_landmarks[0][12].y,
            pose_result.pose_landmarks[0][12].z
        ])
        
        for idx in pose_indices:
            lm = pose_result.pose_landmarks[0][idx]
            point = np.array([lm.x, lm.y, lm.z])
            face_dist = np.linalg.norm(point - face_center)
            left_dist = np.linalg.norm(point - left_shoulder)
            right_dist = np.linalg.norm(point - right_shoulder)
            shoulder_dist = min(left_dist, right_dist)
            rel_x = point[0] - face_center[0]
            rel_y = point[1] - face_center[1]
            rel_z = point[2] - face_center[2]
            features.extend([rel_x, rel_y, rel_z, face_dist, shoulder_dist])
    else:
        features.extend([0.0] * (6 * 5))
    
    # Face
    if face_result.face_landmarks:
        face_indices = [33, 263, 1, 61, 291]
        for idx in face_indices:
            lm = face_result.face_landmarks[0][idx]
            point = np.array([lm.x, lm.y, lm.z])
            rel_x = point[0] - face_center[0]
            rel_y = point[1] - face_center[1]
            rel_z = point[2] - face_center[2]
            dist = np.linalg.norm(point - face_center)
            features.extend([rel_x, rel_y, rel_z, dist])
    else:
        features.extend([0.0] * (5 * 4))
    
    return np.array(features)

def find_matches(features, top_k=5):
    """Find top K matches using cosine similarity"""
    if features is None:
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
async def recognize_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Recognize sign language from uploaded image
    
    - **file**: Image file (jpg, png, etc.)
    - **top_k**: Number of top predictions to return (default: 5)
    """
    import time
    start = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract features
        features = extract_features(frame)
        
        if features is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Find matches
        predictions = find_matches(features, top_k)
        
        processing_time = (time.time() - start) * 1000
        
        return RecognitionResponse(
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize/landmarks", response_model=RecognitionResponse)
async def recognize_landmarks(request: LandmarkRequest):
    """
    Recognize sign language from landmark vector
    
    - **vector**: 260D landmark vector
    - **top_k**: Number of top predictions to return (default: 5)
    """
    import time
    start = time.time()
    
    try:
        if len(request.vector) != 260:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 260D vector, got {len(request.vector)}D"
            )
        
        features = np.array(request.vector)
        
        # Find matches
        predictions = find_matches(features, request.top_k)
        
        processing_time = (time.time() - start) * 1000
        
        return RecognitionResponse(
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectors_loaded": len(model_state.vectors) if model_state.vectors is not None else 0
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
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
