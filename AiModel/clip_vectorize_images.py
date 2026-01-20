import cv2
import torch
from PIL import Image
import numpy as np
import requests
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModel
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

load_dotenv()


class CLIPImageVectorizer:
    def __init__(self, model_name: str = "google/siglip-large-patch16-384"):
        print("Loading vision model...")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
        # MediaPipe Tasks API
        if MEDIAPIPE_AVAILABLE:
            try:
                hand_base = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
                hand_opts = vision.HandLandmarkerOptions(base_options=hand_base, num_hands=2)
                self.hands = vision.HandLandmarker.create_from_options(hand_opts)
                
                pose_base = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
                pose_opts = vision.PoseLandmarkerOptions(base_options=pose_base)
                self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
                print("✓ MediaPipe Tasks API initialized")
            except:
                self.hands = None
                self.pose = None
                print("⚠ MediaPipe models not found - using center crop")
        else:
            self.hands = None
            self.pose = None
            print("⚠ MediaPipe not available - using center crop")
        
        raw_url = os.getenv('q_url', 'http://localhost:6333')
        if 'cloud.qdrant.io' in raw_url:
            if not raw_url.startswith('http'):
                raw_url = f"https://{raw_url}"
            raw_url = raw_url.replace('https://', '').replace('http://', '').rstrip(':6333').rstrip(':443')
            self.qdrant_url = f"https://{raw_url}:6333"
        else:
            self.qdrant_url = raw_url.rstrip('/')
        
        self.api_key = os.getenv('q_api', '')
        self.headers = {'Content-Type': 'application/json'}
        if self.api_key:
            self.headers['api-key'] = self.api_key
        
        print(f"✓ Qdrant: {self.qdrant_url}\n")
    
    def crop_upper_body(self, img):
        """Crop to upper body with MediaPipe Tasks API"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        if not self.hands or not self.pose:
            crop_h = int(h * 0.6)
            start_y = (h - crop_h) // 2
            return Image.fromarray(img_array[start_y:start_y + crop_h, :])
        
        # Convert to MediaPipe Image
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        hand_results = self.hands.detect(mp_img)
        pose_results = self.pose.detect(mp_img)
        
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
            return img
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        return Image.fromarray(img_array[y_min:y_max, x_min:x_max])
    
    def extract_landmarks(self, img):
        """Extract normalized 138D landmarks"""
        if not self.hands or not self.pose:
            return np.zeros(138)
        
        img_array = np.array(img)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        hand_results = self.hands.detect(mp_img)
        pose_results = self.pose.detect(mp_img)
        
        landmarks = np.zeros(138)
        
        # Normalize hand landmarks relative to wrist (landmark 0)
        if hand_results.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.hand_landmarks[:2]):
                offset = i * 63
                wrist = hand_landmarks[0]  # Wrist is landmark 0
                for j, lm in enumerate(hand_landmarks):
                    landmarks[offset + j*3] = lm.x - wrist.x
                    landmarks[offset + j*3 + 1] = lm.y - wrist.y
                    landmarks[offset + j*3 + 2] = lm.z - wrist.z
        
        # Normalize pose landmarks (shoulders/elbows) relative to elbow midpoint
        if pose_results.pose_landmarks:
            for landmarks_list in pose_results.pose_landmarks:
                left_elbow = landmarks_list[13]   # Left elbow
                right_elbow = landmarks_list[14]  # Right elbow
                elbow_center_x = (left_elbow.x + right_elbow.x) / 2
                elbow_center_y = (left_elbow.y + right_elbow.y) / 2
                elbow_center_z = (left_elbow.z + right_elbow.z) / 2
                
                for i, idx in enumerate([11, 12, 13, 14]):
                    lm = landmarks_list[idx]
                    landmarks[126 + i*3] = lm.x - elbow_center_x
                    landmarks[126 + i*3 + 1] = lm.y - elbow_center_y
                    landmarks[126 + i*3 + 2] = lm.z - elbow_center_z
        
        return landmarks
    
    def encode_images_batch(self, img_paths, batch_size=8):
        """Encode multiple images in batches"""
        img_vectors = []
        landmark_vectors = []
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            batch_imgs = []
            batch_landmarks = []
            
            for img_path in batch_paths:
                img = Image.open(img_path).convert('RGB')
                cropped = self.crop_upper_body(img)
                batch_imgs.append(cropped.resize((384, 384), Image.Resampling.LANCZOS))
                batch_landmarks.append(self.extract_landmarks(img))
            
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            
            img_vectors.extend(features.cpu().numpy())
            landmark_vectors.extend(batch_landmarks)
        
        return img_vectors, landmark_vectors
    
    def upload_batch_to_qdrant(self, img_vectors, landmark_vectors, labels, sources):
        """Upload multiple vectors at once"""
        img_points = []
        landmark_points = []
        
        for img_vec, landmark_vec, label, source in zip(img_vectors, landmark_vectors, labels, sources):
            point_id = abs(hash(f"{source}_{label}")) % (10**8)
            
            img_points.append({
                "id": point_id,
                "vector": img_vec.tolist(),
                "payload": {"label": label, "source": source, "frames": 1}
            })
            
            landmark_points.append({
                "id": point_id,
                "vector": landmark_vec.tolist(),
                "payload": {"label": label, "source": source, "frames": 1}
            })
        
        r1 = requests.put(
            f"{self.qdrant_url}/collections/clip_videos/points",
            json={"points": img_points},
            headers=self.headers,
            timeout=30
        )
        
        r2 = requests.put(
            f"{self.qdrant_url}/collections/hand_landmarks/points",
            json={"points": landmark_points},
            headers=self.headers,
            timeout=30
        )
        
        if r1.status_code in [200, 201] and r2.status_code in [200, 201]:
            return len(img_points)
        return 0
    
    def process_directory(self, directory: Path, batch_size=16):
        """Process all images in subdirectories, using folder name as label"""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"No subdirectories found in {directory}")
            return
        
        print(f"Found {len(subdirs)} label folders\n")
        
        for subdir in subdirs:
            label = subdir.name.lower()
            images = [f for f in subdir.iterdir() if f.suffix.lower() in image_exts][:1200]  # Max 200 per folder
            
            if not images:
                continue
            
            print(f"Processing {label}: {len(images)} images")
            
            try:
                # Batch encode
                img_vecs, landmark_vecs = self.encode_images_batch(images, batch_size)
                labels = [label] * len(images)
                sources = [str(p) for p in images]
                
                # Batch upload
                uploaded = self.upload_batch_to_qdrant(img_vecs, landmark_vecs, labels, sources)
                print(f"  ✓ Uploaded {uploaded} images")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print("\n✓ Complete")
    
    def process_single(self, img_path: Path, label: str):
        print(f"Processing: {img_path.name}")
        img_vecs, landmark_vecs = self.encode_images_batch([img_path], batch_size=1)
        uploaded = self.upload_batch_to_qdrant(img_vecs, landmark_vecs, [label], [str(img_path)])
        print(f"✓ Uploaded {uploaded} image")


def main():
    parser = argparse.ArgumentParser(description='CLIP Image Vectorization')
    parser.add_argument('--input', required=True, help='Directory with label folders or single image')
    parser.add_argument('--label', help='Label (required for single image, auto from folder otherwise)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for encoding (default: 16)')
    
    args = parser.parse_args()
    
    vectorizer = CLIPImageVectorizer()
    input_path = Path(args.input)
    
    if input_path.is_dir():
        vectorizer.process_directory(input_path, args.batch_size)
    elif input_path.is_file():
        if not args.label:
            # Use parent folder name as label
            args.label = input_path.parent.name.lower()
        vectorizer.process_single(input_path, args.label)
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()