import cv2
import numpy as np
import json
import mediapipe as mp
from pathlib import Path
from collections import deque

class LiveSignRecognizer:
    def __init__(self, vectors_dir="vectors", video_mode=False):
        # Load MediaPipe
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
                base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5
            )
        )
        self.pose = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
                running_mode=VisionRunningMode.IMAGE,
                min_pose_detection_confidence=0.5
            )
        )
        self.face = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5
            )
        )
        
        # Load all vectors
        print("Loading vectors...")
        self.vectors = []
        self.labels = []
        
        vectors_path = Path(vectors_dir)
        for json_file in vectors_path.rglob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
                self.vectors.append(np.array(data["vector"]))
                self.labels.append(data["label"])
        
        self.vectors = np.array(self.vectors)
        print(f"Loaded {len(self.vectors)} vectors")
        
        # Smoothing
        self.prediction_history = deque(maxlen=5)
        self.video_mode = video_mode
    
    def extract_features(self, frame):
        """Extract normalized features from frame"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        hands_result = self.hands.detect(mp_image)
        pose_result = self.pose.detect(mp_image)
        face_result = self.face.detect(mp_image)
        
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
    
    def find_match(self, features):
        """Find closest match using cosine similarity"""
        if features is None:
            return None, 0.0
        
        # Normalize
        features_norm = features / (np.linalg.norm(features) + 1e-8)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = np.dot(vectors_norm, features_norm)
        best_idx = np.argmax(similarities)
        
        return self.labels[best_idx], similarities[best_idx]
    
    def run(self, video_path=None):
        """Run live recognition from webcam or video file"""
        if video_path:
            cap = cv2.VideoCapture(video_path)
            print(f"Processing video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("Starting live recognition from webcam...")
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract features
            features = self.extract_features(frame)
            
            # Find match
            label, confidence = self.find_match(features)
            
            # Smooth predictions
            if label:
                self.prediction_history.append((label, confidence))
                
                # Get most common prediction
                if len(self.prediction_history) >= 3:
                    labels_only = [l for l, c in self.prediction_history if c > 0.7]
                    if labels_only:
                        from collections import Counter
                        most_common = Counter(labels_only).most_common(1)[0][0]
                        avg_conf = np.mean([c for l, c in self.prediction_history if l == most_common])
                        label = most_common
                        confidence = avg_conf
            
            # Display
            display_frame = frame.copy()
            
            if label and confidence > 0.6:
                # Draw prediction
                text = f"{label}: {confidence:.2%}"
                cv2.putText(display_frame, text, (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                # Confidence bar
                bar_width = int(confidence * 400)
                cv2.rectangle(display_frame, (20, 100), (20 + bar_width, 130), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (20, 100), (420, 130), (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No match", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            # Instructions
            cv2.putText(display_frame, "Press 'q' to quit", (20, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Sign Language Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    video_path = None
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        recognizer = LiveSignRecognizer(video_mode=True)
        recognizer.run(video_path)
    else:
        recognizer = LiveSignRecognizer()
        recognizer.run()
