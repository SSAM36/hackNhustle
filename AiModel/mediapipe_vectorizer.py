import os
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

class MediaPipeVectorizer:
    def __init__(self, model_dir="models"):
        # Use custom task files for better accuracy
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        self.hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=f"{model_dir}/hand_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5
            )
        )
        self.pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=f"{model_dir}/pose_landmarker_lite.task"),
                running_mode=VisionRunningMode.IMAGE,
                min_pose_detection_confidence=0.5
            )
        )
        self.face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=f"{model_dir}/face_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
        )
    
    def extract_normalized_features(self, image_path, mirror=False):
        """Extract and normalize all landmarks with ratios"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Mirror if requested
        if mirror:
            img_rgb = cv2.flip(img_rgb, 1)
        
        h, w = img_rgb.shape[:2]
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Get all landmarks
        hands_result = self.hands.detect(mp_image)
        pose_result = self.pose.detect(mp_image)
        face_result = self.face.detect(mp_image)
        
        # Extract face center (nose tip) as reference
        face_center = None
        if face_result.face_landmarks:
            nose = face_result.face_landmarks[0][1]  # nose tip
            face_center = np.array([nose.x, nose.y, nose.z])
        
        if face_center is None:
            return None
        
        features = []
        
        # Process hands
        if hands_result.hand_landmarks:
            for hand_landmarks in hands_result.hand_landmarks:
                wrist = np.array([
                    hand_landmarks[0].x,
                    hand_landmarks[0].y,
                    hand_landmarks[0].z
                ])
                
                # Normalize hand points relative to wrist
                for i in range(21):
                    lm = hand_landmarks[i]
                    point = np.array([lm.x, lm.y, lm.z])
                    
                    # Distance from wrist (normalized)
                    wrist_dist = np.linalg.norm(point - wrist)
                    
                    # Distance from face (normalized)
                    face_dist = np.linalg.norm(point - face_center)
                    
                    # Relative position to wrist
                    rel_x = point[0] - wrist[0]
                    rel_y = point[1] - wrist[1]
                    rel_z = point[2] - wrist[2]
                    
                    features.extend([rel_x, rel_y, rel_z, wrist_dist, face_dist])
        else:
            # No hands detected - pad with zeros
            features.extend([0.0] * (21 * 5 * 2))  # 2 hands max
        
        # Pad if only one hand
        if hands_result.hand_landmarks and len(hands_result.hand_landmarks) == 1:
            features.extend([0.0] * (21 * 5))
        
        # Process pose (shoulders, elbows, wrists)
        if pose_result.pose_landmarks:
            # Key pose points: shoulders (11,12), elbows (13,14), wrists (15,16)
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
                
                # Distance from face
                face_dist = np.linalg.norm(point - face_center)
                
                # Distance from nearest shoulder
                left_dist = np.linalg.norm(point - left_shoulder)
                right_dist = np.linalg.norm(point - right_shoulder)
                shoulder_dist = min(left_dist, right_dist)
                
                # Relative to face center
                rel_x = point[0] - face_center[0]
                rel_y = point[1] - face_center[1]
                rel_z = point[2] - face_center[2]
                
                features.extend([rel_x, rel_y, rel_z, face_dist, shoulder_dist])
        else:
            features.extend([0.0] * (6 * 5))
        
        # Process face key points (eyes, nose, mouth corners)
        if face_result.face_landmarks:
            # Key face indices: left eye (33), right eye (263), nose (1), mouth left (61), mouth right (291)
            face_indices = [33, 263, 1, 61, 291]
            
            for idx in face_indices:
                lm = face_result.face_landmarks[0][idx]
                point = np.array([lm.x, lm.y, lm.z])
                
                # Relative to face center
                rel_x = point[0] - face_center[0]
                rel_y = point[1] - face_center[1]
                rel_z = point[2] - face_center[2]
                
                # Distance from center
                dist = np.linalg.norm(point - face_center)
                
                features.extend([rel_x, rel_y, rel_z, dist])
        else:
            features.extend([0.0] * (5 * 4))
        
        return features
    
    def process_folder(self, data_dir, output_dir):
        """Process all images in data_dir and save vectors to output_dir"""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Support both flat structure (data/a.jpeg) and nested (data/A/img1.jpeg)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(data_path.glob(ext))
            image_files.extend(data_path.glob(f'*/{ext}'))
            image_files.extend(data_path.glob(f'*/*/{ext}'))
        
        print(f"Found {len(image_files)} images")
        
        processed = 0
        failed = 0
        
        for img_path in image_files:
            try:
                # Extract features (original)
                features = self.extract_normalized_features(img_path, mirror=False)
                
                if features is None:
                    failed += 1
                    print(f"Failed: {img_path.name} (no face detected)")
                    continue
                
                # Extract features (mirrored)
                features_mirror = self.extract_normalized_features(img_path, mirror=True)
                
                # Create output path maintaining structure
                rel_path = img_path.relative_to(data_path)
                output_file = output_path / rel_path.parent / f"{img_path.stem}.json"
                output_file_mirror = output_path / rel_path.parent / f"{img_path.stem}_mirror.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save original vector
                vector_data = {
                    "file": str(img_path),
                    "label": img_path.parent.name if img_path.parent != data_path else img_path.stem,
                    "vector": features,
                    "dimension": len(features),
                    "augmentation": "original"
                }
                
                with open(output_file, 'w') as f:
                    json.dump(vector_data, f, indent=2)
                
                # Save mirrored vector
                if features_mirror is not None:
                    vector_data_mirror = {
                        "file": str(img_path),
                        "label": img_path.parent.name if img_path.parent != data_path else img_path.stem,
                        "vector": features_mirror,
                        "dimension": len(features_mirror),
                        "augmentation": "mirror"
                    }
                    
                    with open(output_file_mirror, 'w') as f:
                        json.dump(vector_data_mirror, f, indent=2)
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed: {processed}, Failed: {failed}")
                    
            except Exception as e:
                failed += 1
                print(f"Error processing {img_path.name}: {e}")
        
        print(f"\nComplete! Processed: {processed}, Failed: {failed}")
        print(f"Vectors saved to: {output_path}")

if __name__ == "__main__":
    vectorizer = MediaPipeVectorizer()
    vectorizer.process_folder("data", "vectors")
