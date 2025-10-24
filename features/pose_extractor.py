"""
Module 1: Pose Extractor
Handle human-pose inference using YOLOv8-Pose or MediaPipe Pose.
"""

from typing import Any
import numpy as np
from ultralytics import YOLO


def load_pose_model(model_type: str = "yolo") -> Any:
    """
    Load pose-estimation model (YOLOv8n-pose or MediaPipe).
    
    Args:
        model_type: Type of model to load ("yolo" or "mediapipe")
    
    Returns:
        Loaded pose estimation model
    """
    if model_type.lower() == "yolo":
        # Load YOLOv8n-pose model
        model = YOLO("yolov8n-pose.pt")
        return model
    elif model_type.lower() == "mediapipe":
        # MediaPipe implementation can be added later
        raise NotImplementedError("MediaPipe support not yet implemented")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_keypoints(model: Any, frame: np.ndarray) -> list[dict]:
    """
    Run inference and return list of detected persons.
    
    Args:
        model: Loaded pose estimation model
        frame: Input frame (numpy array)
    
    Returns:
        List of detected persons, each containing:
        {
            'id': int,
            'keypoints': {
                'nose': (x, y),
                'shoulder_l': (x, y),
                'shoulder_r': (x, y),
                'hip_l': (x, y),
                'hip_r': (x, y),
                'knee_l': (x, y),
                'knee_r': (x, y),
                'ankle_l': (x, y),
                'ankle_r': (x, y)
            },
            'conf': float,
            'bbox': (x1, y1, x2, y2)
        }
    """
    # Run inference
    results = model(frame, verbose=False)
    
    persons = []
    
    # Process each detection
    for idx, result in enumerate(results):
        if result.keypoints is None or len(result.keypoints) == 0:
            continue
            
        # Get boxes and keypoints
        boxes = result.boxes
        keypoints = result.keypoints
        
        for person_idx in range(len(boxes)):
            # Get bounding box
            box = boxes[person_idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            # Get keypoints (COCO format: 17 keypoints)
            kps = keypoints[person_idx].xy[0].cpu().numpy()  # Shape: (17, 2)
            
            # YOLOv8-pose uses COCO keypoint format:
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
            # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
            # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
            # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
            
            keypoints_dict = {
                'nose': tuple(kps[0]),
                'shoulder_l': tuple(kps[5]),
                'shoulder_r': tuple(kps[6]),
                'hip_l': tuple(kps[11]),
                'hip_r': tuple(kps[12]),
                'knee_l': tuple(kps[13]),
                'knee_r': tuple(kps[14]),
                'ankle_l': tuple(kps[15]),
                'ankle_r': tuple(kps[16])
            }
            
            person_data = {
                'id': person_idx,
                'keypoints': keypoints_dict,
                'conf': conf,
                'bbox': (float(x1), float(y1), float(x2), float(y2))
            }
            
            persons.append(person_data)
    
    return persons
