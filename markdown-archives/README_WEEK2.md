# Week 2: Pose Feature Extraction Pipeline

## 📋 Overview

This week's implementation focuses on extracting geometric features from human poses using YOLOv8n-pose for person Re-ID.

## 🏗️ Project Structure

```
yolo/
├── features/
│   ├── __init__.py
│   ├── pose_extractor.py      # Module 1: YOLOv8n-pose inference
│   ├── ratio_calculator.py     # Module 2: Geometric descriptors
│   ├── feature_logger.py       # Module 3: CSV logging
│   └── visualize_pose.py       # Module 4: Skeleton rendering
├── main_features.py            # Module 5: Integration pipeline
├── outputs/
│   ├── features.csv           # Per-frame geometric dataset
│   └── annotated_video.mp4    # Pose + metrics overlay
└── videos/
    └── test_scene.MOV         # Input test video
```

## 🔧 Modules

### Module 1: `pose_extractor.py`
- **Purpose**: Handle YOLOv8n-pose inference
- **Functions**:
  - `load_pose_model(model_type="yolo")`: Load pose model
  - `get_keypoints(model, frame)`: Extract keypoints from frame
- **Output**: List of detected persons with keypoints, confidence, and bbox

### Module 2: `ratio_calculator.py`
- **Purpose**: Compute geometric descriptors
- **Functions**:
  - `estimate_height_px(kps)`: Pixel height from nose to ankle
  - `calc_shoulder_hip_ratio(kps)`: Shoulder width / hip width
  - `calc_torso_leg_ratio(kps)`: Torso length / leg length
  - `compile_features(kps, bbox)`: Combine all metrics
- **Output**: Feature dictionary with height, ratios, and bbox center

### Module 3: `feature_logger.py`
- **Purpose**: Save per-frame metrics to CSV
- **Functions**:
  - `init_logger(csv_path)`: Create CSV with headers
  - `log_features(frame_idx, features, conf, csv_path)`: Append row
- **CSV Columns**: `frame, height_px, shoulder_hip, torso_leg, x_center, y_center, conf`

### Module 4: `visualize_pose.py`
- **Purpose**: Render skeleton and metrics on frames
- **Functions**:
  - `draw_pose(frame, kps)`: Draw keypoint skeleton
  - `overlay_metrics(frame, features)`: Draw feature text
- **Visual Output**: Skeleton lines + metric text overlay

### Module 5: `main_features.py`
- **Purpose**: End-to-end integration pipeline
- **Features**:
  - Video processing with real-time display
  - CSV logging of all features
  - Annotated video output
  - FPS monitoring and progress tracking

## 🚀 Usage

### Run the pipeline:

```bash
python main_features.py
```

### Expected behavior:
1. Loads YOLOv8n-pose model (auto-downloads if not present)
2. Processes `videos/test_scene.MOV`
3. Displays real-time annotated video
4. Saves features to `outputs/features.csv`
5. Saves annotated video to `outputs/annotated_video.mp4`
6. Press 'q' to quit early

## 📊 Output Examples

### CSV Data (`outputs/features.csv`)
```csv
frame,height_px,shoulder_hip,torso_leg,x_center,y_center,conf
0,320.50,1.24,0.83,210.00,230.00,0.92
1,322.10,1.26,0.81,212.50,228.00,0.94
```

### Keypoints Dictionary
```python
{
    'id': 0,
    'keypoints': {
        'nose': (210, 60),
        'shoulder_l': (170, 120),
        'shoulder_r': (250, 120),
        'hip_l': (180, 240),
        'hip_r': (240, 240),
        'knee_l': (175, 340),
        'knee_r': (245, 340),
        'ankle_l': (170, 420),
        'ankle_r': (250, 420)
    },
    'conf': 0.92,
    'bbox': (160, 40, 260, 420)
}
```

## 🎯 Week 2 Deliverables

| Deliverable              | Status | Location                    |
| ------------------------ | ------ | --------------------------- |
| `main_features.py`       | ✅      | Root directory              |
| `outputs/features.csv`   | ✅      | Generated on run            |
| Annotated video          | ✅      | `outputs/annotated_video.mp4` |
| Metrics stability report | 🔄      | Manual analysis required    |
| FPS target (≥ 15 FPS)    | 🔄      | Verify on Mac CPU           |

## 🧪 Testing Checklist

- [ ] YOLOv8n-pose model loads successfully
- [ ] Video opens and processes frames
- [ ] Keypoints are detected and extracted
- [ ] Features are calculated correctly
- [ ] CSV is created and populated
- [ ] Skeleton is drawn on frames
- [ ] Metrics overlay is visible
- [ ] Output video is saved
- [ ] FPS is ≥ 15 on Mac CPU
- [ ] Feature stability < 10% variation for same subject

## 🔍 Key Features

- **9 critical keypoints** tracked: nose, shoulders, hips, knees, ankles
- **3 geometric ratios** computed for Re-ID
- **Real-time visualization** with skeleton and metrics
- **CSV logging** for analysis and model training
- **Progress tracking** with FPS monitoring

## 📝 Notes

- YOLOv8n-pose uses COCO 17-keypoint format
- Model will auto-download on first run (~6MB)
- Ensure `videos/test_scene.MOV` exists
- Press 'q' to stop processing early
- Output video uses mp4v codec for compatibility
