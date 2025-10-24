

---

# 🧩 Week 4 – Appearance-Enhanced Re-ID + Stabilization

## 🎯 Objective

Increase ID stability and reduce false splits by fusing **geometric, color, and texture features** with motion-assisted matching and smoothed ratio inputs.
This replaces noisy pseudo-depth dependence with richer 2-D appearance cues.

---

## 🧠 Overview

Current issues: ID switching ≈ 70 %, distance variance ≈ 30 %.
Goal: reach ≥ 80 % ID retention and ≤ 15 % variance by combining:

1. **Motion-assisted assignment** (IoU + center proximity)
2. **Smoothed torso-leg ratio** for pseudo-depth stability
3. **Color & texture descriptors** for visual identity
4. **Feature normalization + weighted fusion**

---

## 🧩 Modules

### 1️⃣ `appearance_extractor.py`  (new)

Extracts dominant clothing colors + texture cues.

```python
import cv2, numpy as np

def extract_appearance_features(frame, bbox):
    """Return dict with top/bottom HSV mean and texture variance."""
    x1,y1,x2,y2 = map(int, bbox)
    person = frame[y1:y2, x1:x2]
    h, w = person.shape[:2]
    top = person[:int(0.6*h), :]
    bottom = person[int(0.6*h):, :]

    hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
    top_hsv = np.mean(hsv[:int(0.6*h), :, 0])
    bot_hsv = np.mean(hsv[int(0.6*h):, :, 0])
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0

    return {"top_h": top_hsv/180.0, "bot_h": bot_hsv/180.0, "texture": texture}
```

---

### 2️⃣ `reid_tracker.py`  (update)

Add appearance + motion fusion to cost:

```python
cost = (
  0.30*ΔHnorm +
  0.20*ΔRsh +
  0.10*ΔRtl +
  0.25*color_distance(top/bot) +
  0.05*Δtexture +
  0.10*motion_cost(IOU, center)
)
gate = 0.45
```

* **Motion term** = `0.5*(1 – IOU) + 0.5*(center_dist/img_diag)`
* Apply 5-frame EMA on torso-leg ratio before cost computation.

---

### 3️⃣ `filter_smooth.py`  (update)

Add temporal smoothing for ratios & color values:

```python
alpha=0.3
r_tl_smooth = alpha*curr + (1-alpha)*prev
```

---

### 4️⃣ `fusion_logger.py`

Log extended features:

```
frame,track_id,height_px,height_norm,shoulder_hip,torso_leg,
top_h,bot_h,texture,conf
```

---

### 5️⃣ `visualizer_bbox.py`

Show bounding boxes + ID + dominant color hue (small patch beside ID text).

---

## 📊 Evaluation Metrics

| Metric                 | Target   | Notes                         |
| ---------------------- | -------- | ----------------------------- |
| ID retention rate      | ≥ 80 %   | Same ID across frames         |
| Avg distance variance  | ≤ 15 %   | Smoother pseudo-depth         |
| Height stability (std) | ≤ 0.15 m | Should remain consistent      |
| FPS                    | ≥ 16 FPS | Maintain realtime performance |

---

## ⚙️ Deliverables

* `outputs/reid_appearance.csv` — logs with color + texture
* `outputs/reid_overlay_v2.mp4` — bbox visualization with ID + color cues
* `outputs/reid_summary_v2.txt` — per-ID stability + variance report

---

## 🚀 Next Step Preview (Week 5)

Add **gait estimation** and prepare final **Re-ID fusion weighting** for real LiDAR integration.

---
