## 🧩 MODULE 1 — `pose_extractor.py`

### 🎯 Purpose

Handle human-pose inference for each frame using either **YOLOv8-Pose** or **MediaPipe Pose**.

### 🔧 Functions

```python
def load_pose_model(model_type="yolo") -> Any:
    """Load pose-estimation model (YOLOv8n-pose or MediaPipe)."""
```

```python
def get_keypoints(model, frame: np.ndarray) -> list[dict]:
    """
    Run inference and return list of detected persons.
    Each dict = {
        'id': int,
        'keypoints': { 'nose':(x,y), 'shoulder_l':(x,y), 'shoulder_r':(x,y),
                       'hip_l':(x,y), 'hip_r':(x,y),
                       'knee_l':(x,y), 'knee_r':(x,y),
                       'ankle_l':(x,y), 'ankle_r':(x,y) },
        'conf': float,
        'bbox': (x1,y1,x2,y2)
    }
    """
```

### 🧪 Output Example

```python
[
  {'id':0,'keypoints':{'nose':(210,60),'shoulder_l':(170,120),...},'conf':0.92,'bbox':(160,40,260,420)}
]
```

---

## 🧩 MODULE 2 — `ratio_calculator.py`

### 🎯 Purpose

Compute geometric descriptors from pose keypoints.

### 🔧 Functions

```python
def estimate_height_px(kps: dict) -> float:
    """Return pixel height = y(ankle_avg) - y(nose)."""

def calc_shoulder_hip_ratio(kps: dict) -> float:
    """Return width(shoulder)/width(hip)."""

def calc_torso_leg_ratio(kps: dict) -> float:
    """Return vertical distance(shoulder→hip) / (hip→ankle)."""

def compile_features(kps: dict, bbox: tuple) -> dict:
    """Combine all metrics into one feature vector."""
```

### 🧪 Output Example

```python
{
  'height_px': 320.5,
  'shoulder_hip': 1.24,
  'torso_leg': 0.83,
  'bbox_center': (210,230)
}
```

---

## 🧩 MODULE 3 — `feature_logger.py`

### 🎯 Purpose

Save per-frame metrics for analysis and later Re-ID training.

### 🔧 Functions

```python
def init_logger(csv_path="outputs/features.csv") -> None:
    """Create CSV header if not exists."""

def log_features(frame_idx:int, features:dict, conf:float, csv_path:str) -> None:
    """Append one row of data."""
```

### 🧪 CSV Columns

```
frame,height_px,shoulder_hip,torso_leg,x_center,y_center,conf
```

---

## 🧩 MODULE 4 — `visualize_pose.py`

### 🎯 Purpose

Render skeleton and feature text on each frame for debugging.

### 🔧 Functions

```python
def draw_pose(frame: np.ndarray, kps: dict) -> np.ndarray:
    """Draw keypoint skeleton."""

def overlay_metrics(frame: np.ndarray, features: dict) -> np.ndarray:
    """Draw height/ratio text on image."""
```

### 🧪 Visual Output

* Skeleton lines (shoulder→hip→knee→ankle)
* Text: “H: 320  SH: 1.24  TL: 0.83”

---

## 🧩 MODULE 5 — Integration (`main_features.py`)

### 🎯 Pipeline Pseudocode

```python
from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.feature_logger import init_logger, log_features
from features.visualize_pose import draw_pose, overlay_metrics
import cv2, time

model = load_pose_model("yolo")
init_logger()

cap = cv2.VideoCapture("videos/test_scene.MOV")
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    persons = get_keypoints(model, frame)
    for p in persons:
        feats = compile_features(p["keypoints"], p["bbox"])
        log_features(frame_id, feats, p["conf"], "outputs/features.csv")
        frame = draw_pose(frame, p["keypoints"])
        frame = overlay_metrics(frame, feats)

    cv2.imshow("Pose Features", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    frame_id += 1
```

---

## 📊 Expected Results (End of Week 2)

| Deliverable              | Description                        |
| ------------------------ | ---------------------------------- |
| `main_features.py`       | End-to-end working video processor |
| `outputs/features.csv`   | Per-frame geometric dataset        |
| Annotated video          | Pose + metrics overlay             |
| Metrics stability report | Variation < 10% for same subject   |
| FPS target               | ≥ 15 FPS (Mac CPU)                 |

---
