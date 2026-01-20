"""
CLIP CAMERA SEARCH
==================
Captures frames, encodes with CLIP, searches Qdrant.

Usage:
    python3 clip_camera_search.py
    python3 clip_camera_search.py --collection clip_videos

Controls:
    SPACE - Capture and search
    q - Quit
"""

import cv2
import torch
from PIL import Image
import numpy as np

import requests
import argparse
import os
from dotenv import load_dotenv
import time
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from threading import Thread
from queue import Queue

load_dotenv()


class CLIPCameraSearch:
    def __init__(self, model_name: str = "google/siglip-large-patch16-384",
                 qdrant_url: str = None, api_key: str = None,
                 collection: str = "clip_videos", buffer_seconds: int = 3, fps: int = 10):
        
        print("Loading vision model...")
        if 'siglip' in model_name.lower():
            self.model = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded: {model_name.split('/')[-1]} on {self.device}")
        
        raw_url = qdrant_url or os.getenv('q_url')
        raw_url = raw_url.rstrip('/')

        if 'cloud.qdrant.io' in raw_url:
            # Qdrant Cloud → HTTPS only, NO port
            if not raw_url.startswith('http'):
                raw_url = f"https://{raw_url}"
            self.qdrant_url = raw_url
        else:
            # Local Qdrant
            self.qdrant_url = raw_url

        self.api_key = api_key or os.getenv('q_api', '')
        self.collection = collection

        self.headers = {'Content-Type': 'application/json'}
        if self.api_key:
            self.headers['api-key'] = self.api_key

        self.buffer_size = buffer_seconds * fps
        self.frame_buffer = []
        self.last_search_time = 0
        self.search_interval = 0.2
        self.qdrant_available = False

        # Search results
        self.last_results = []
        self.search_in_progress = False

        # Test Qdrant connection
        try:
            r = requests.get(
                f"{self.qdrant_url}/collections/{self.collection}",
                headers=self.headers,
                timeout=5
            )
            self.qdrant_available = r.status_code == 200
            if self.qdrant_available:
                print(f"✓ Qdrant connected at {self.qdrant_url}")
            else:
                print(f"✗ Qdrant responded with status {r.status_code}")
        except Exception as e:
            print(f"✗ Qdrant connection failed: {e}")

        print(f"✓ Collection: {self.collection}")
        print(f"✓ Buffer: {buffer_seconds}s ({self.buffer_size} frames)")
        if not self.qdrant_available:
            print("⚠ Qdrant unavailable - search disabled\n")
        else:
            print()

    def crop_horizontal(self, frame):
        h, w = frame.shape[:2]
        crop_h = int(h * 0.6)
        start_y = (h - crop_h) // 2
        return frame[start_y:start_y + crop_h, :]
    
    def encode_frame(self, frame):
        """Encode a single frame to CLIP embedding with enhanced preprocessing"""
        try:
            # Convert BGR to RGB properly
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply histogram equalization for better contrast
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Slight denoising while preserving edges
            rgb = cv2.bilateralFilter(rgb, 5, 50, 50)
            
            # Convert to PIL and resize with high-quality resampling
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.resize((384, 384), Image.Resampling.LANCZOS)
            
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Encoding error: {e}")
            return None
    
    def search_qdrant(self, vector, top_k: int = 15):
        """Search Qdrant for similar vectors with score filtering"""
        if not self.qdrant_available:
            return []
        
        try:
            r = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/search",
                json={"vector": vector.tolist(), "limit": top_k, "with_payload": True, "score_threshold": 0.3},
                headers=self.headers,
                timeout=3
            )
            
            if r.status_code == 200:
                results = r.json().get('result', [])
                # Filter and normalize scores for better ranking
                filtered = [(hit["payload"]["label"], hit["score"]) for hit in results if hit["score"] > 0.35]
                return filtered
            else:
                print(f"Search failed with status {r.status_code}")
            return []
        
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def search_thread_worker(self):
        """Background thread with optimized EMA smoothing for stable recognition"""
        ema_scores = {}  # Exponential moving average per label
        alpha = 0.25  # Smoothing factor (lower = more stable)
        decay = 0.97  # Confidence decay per frame (higher = longer retention)
        
        while True:
            current_time = time.time()
            
            if (self.qdrant_available and 
                len(self.frame_buffer) >= 5 and 
                not self.search_in_progress and
                current_time - self.last_search_time > self.search_interval):
                
                self.search_in_progress = True
                self.last_search_time = current_time
                
                # Search with weighted buffer (recent frames more important)
                weights = np.linspace(0.7, 1.0, len(self.frame_buffer))
                weights = weights / weights.sum()
                query_vector = np.average(self.frame_buffer, axis=0, weights=weights)
                query_vector = query_vector / np.linalg.norm(query_vector)
                
                results = self.search_qdrant(query_vector, top_k=15)
                
                if results:
                    # Decay all existing scores
                    for label in ema_scores:
                        ema_scores[label] *= decay
                    
                    # Update EMA with new results (exponential position weighting)
                    for rank, (label, score) in enumerate(results):
                        position_weight = np.exp(-rank * 0.2)  # Exponential decay favors top results
                        weighted_score = score * position_weight
                        
                        if label in ema_scores:
                            ema_scores[label] = alpha * weighted_score + (1 - alpha) * ema_scores[label]
                        else:
                            ema_scores[label] = weighted_score * 0.8  # New labels start slightly lower
                    
                    # Remove low confidence labels with adaptive threshold
                    threshold = 0.08 if len(ema_scores) > 3 else 0.05
                    ema_scores = {k: v for k, v in ema_scores.items() if v > threshold}
                    
                    # Get top 3 labels above display threshold
                    if ema_scores:
                        sorted_labels = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
                        self.last_results = [(label, score) for label, score in sorted_labels[:3] if score > 0.25]
                
                self.search_in_progress = False
            
            time.sleep(0.05)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("="*60)
        print("CLIP CAMERA SEARCH")
        print("="*60)
        print("Auto-searching every 0.2s using buffered frames")
        print("Controls: q=quit")
        print("="*60 + "\n")
        
        # Start background search thread
        search_thread = Thread(target=self.search_thread_worker, daemon=True)
        search_thread.start()
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Convert display to grayscale for speed
            display_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
            
            # Encode every 2nd frame for better temporal resolution
            if frame_count % 2 == 0:
                cropped = self.crop_horizontal(frame)
                # Resize to appropriate size for the model
                cropped = cv2.resize(cropped, (384, 230), interpolation=cv2.INTER_AREA)
                vector = self.encode_frame(cropped)
                
                if vector is not None:
                    self.frame_buffer.append(vector)
                    if len(self.frame_buffer) > self.buffer_size:
                        self.frame_buffer.pop(0)
            
            frame_count += 1
            
            # Draw UI
            display = frame.copy()
            h, w = display.shape[:2]
            
            cv2.putText(display, f"Buffer: {len(self.frame_buffer)}/{self.buffer_size} | Q to quit",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(display, f"Collection: {self.collection}",
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
            
            if not self.qdrant_available:
                status = "Qdrant offline"
                color = (0, 0, 255)
            elif len(self.frame_buffer) < 5:
                status = "Buffering..."
                color = (255, 255, 0)
            elif self.search_in_progress:
                status = "Searching..."
                color = (0, 255, 255)
            else:
                status = "Ready"
                color = (0, 255, 0)
            
            cv2.putText(display, status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if self.last_results:
                y = 150
                cv2.putText(display, "Detected:",
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y += 35
                
                for i, (label, score) in enumerate(self.last_results):
                    # Color based on confidence with better thresholds
                    if score > 0.5:
                        color = (0, 255, 0)  # Green - high confidence
                    elif score > 0.35:
                        color = (0, 255, 255)  # Yellow - medium
                    else:
                        color = (0, 165, 255)  # Orange - low
                    
                    text = f"{i+1}. {label}: {score:.3f}"
                    cv2.putText(display, text, (30, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y += 30
            
            cv2.imshow("CLIP Camera Search", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='CLIP Camera Search')
    parser.add_argument('--collection', default='clip_videos')
    parser.add_argument('--model', default='google/siglip-large-patch16-384',
                       help='Model: google/siglip-large-patch16-384 (1024D), google/siglip-base-patch16-224 (768D)')
    parser.add_argument('--buffer', type=int, default=3, help='Buffer seconds (default: 3)')
    parser.add_argument('--fps', type=int, default=10, help='Processing FPS (default: 10)')
    
    args = parser.parse_args()
    
    searcher = CLIPCameraSearch(
        model_name=args.model,
        collection=args.collection,
        buffer_seconds=args.buffer,
        fps=args.fps
    )
    
    searcher.run()


if __name__ == "__main__":
    main()