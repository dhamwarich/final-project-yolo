# 🧩 Week 2.5: Feature Quality Refinement Module

## 🎯 Objective

Clean and stabilize the per-frame geometric data (height / ratios) generated from pose keypoints to prepare for accurate LiDAR fusion and Re-ID training.

---

## 🧠 Overview of Fixes

1. **Outlier filtering** — reject implausible ratios or missing keypoints
2. **Temporal smoothing (EMA)** — reduce frame-to-frame flicker
3. **Per-person text overlay** — show metrics above each skeleton instead of top-left
4. **Summary statistics report** — auto-calculate mean, std, and outlier % after run

---

## 📁 New / Updated Files

```
/features/
 ├─ filter_smooth.py      # new: filtering + EMA logic
 ├─ visualize_pose.py     # updated: draw per-person overlay
 ├─ feature_logger.py     # updated: add summary generation
 └─ main_features_refined.py  # orchestrates the improvements
```

---

## 🧩 MODULE 1 — `filter_smooth.py`

### Functions

```python
def is_valid_frame(features: dict) -> bool:
    """Rejects frames with impossible values.
    Example thresholds:
        height_px > 100 and shoulder_hip < 3 and torso_leg < 3
    """

def smooth_features(curr: dict, prev: dict, alpha: float = 0.2) -> dict:
    """Apply exponential moving average to numeric fields."""
```

### Behavior

* If invalid → return `None` (skip logging)
* Otherwise → return smoothed values

---

## 🧩 MODULE 2 — `visualize_pose.py` (update)

Modify `overlay_metrics()`:

```python
def overlay_metrics(frame: np.ndarray, features: dict, bbox_center: tuple):
    """Draw text near person’s bounding box center."""
```

Text:
`H:215  SH:1.52  TL:3.80`
Color: `(255,255,0)` (cyan/yellow tone)

---

## 🧩 MODULE 3 — `feature_logger.py` (update)

Add a final report function:

```python
def summarize_log(csv_path: str):
    """Compute mean, std, and outlier percentage for each metric.
       Save summary as outputs/summary.txt."""
```

Example output:

```
Frames processed: 940
Valid frames: 872 (92.7%)
Height_px  mean=410.2  std=38.7
SH_ratio   mean=1.31   std=0.14
TL_ratio   mean=0.79   std=0.09
```

---

## 🧩 MODULE 4 — `main_features_refined.py`

### Pipeline pseudocode

```python
from features.pose_extractor import load_pose_model, get_keypoints
from features.ratio_calculator import compile_features
from features.filter_smooth import is_valid_frame, smooth_features
from features.feature_logger import log_features, summarize_log
from features.visualize_pose import draw_pose, overlay_metrics
import cv2

prev_feats = {}
cap = cv2.VideoCapture("videos/test_scene.MOV")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    persons = get_keypoints(model, frame)
    for p in persons:
        feats = compile_features(p["keypoints"], p["bbox"])
        if not is_valid_frame(feats): continue
        feats = smooth_features(feats, prev_feats)
        prev_feats = feats

        log_features(frame_id, feats, p["conf"], "outputs/features_refined.csv")
        frame = draw_pose(frame, p["keypoints"])
        frame = overlay_metrics(frame, feats, feats["bbox_center"])

cap.release()
summarize_log("outputs/features_refined.csv")
```

---

## 📊 Evaluation Targets

| Metric              | Target                        |
| ------------------- | ----------------------------- |
| Outlier rate        | < 10 % of frames              |
| Ratio variance      | Std < 0.1 across sequence     |
| FPS                 | ≥ 18 FPS (Mac CPU)            |
| Overlay readability | One label per detected person |

---

## 🧩 Deliverables

* `outputs/features_refined.csv` – cleaned log
* `outputs/summary.txt` – mean/std/outlier stats
* `outputs/refined_overlay.mp4` – improved visualization
* Documentation note on filtering thresholds & smoothing α

