
# 🧩 Week 5.5 – Color Normalization & Logging

## 🎯 Objective

Stabilize and correctly visualize color features by normalizing LAB/HSV values, masking low-saturation pixels, and logging per-ID shirt/pants colors.
Ensure that color cues accurately reflect real clothing under mixed lighting.

---

## 🧠 Overview

**Problem:** LAB means clustered near gray (a,b ≈ 128).
**Goal:** Extract perceptually distinct shirt/pants colors per person, maintain real-time speed, and verify by logging average colors in CSV.

---

## 🧩 Pipeline Fixes

### 1️⃣ `appearance_extractor.py` update

Use HSV with saturation masking (or normalized LAB).

```python
import cv2, numpy as np

def extract_color_regions(frame, bbox, use_hsv=True):
    x1, y1, x2, y2 = map(int, bbox)
    person = frame[y1:y2, x1:x2]
    h, w = person.shape[:2]
    top, bottom = person[:int(0.6*h), :], person[int(0.6*h):, :]

    if use_hsv:
        hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
        H, S, _ = cv2.split(hsv)
        mask = S > 30
        shirt_hist = cv2.calcHist([H[:int(0.6*h), :]], [0], mask[:int(0.6*h), :], [16], [0,180])
        pants_hist = cv2.calcHist([H[int(0.6*h):, :]], [0], mask[int(0.6*h):, :], [16], [0,180])
    else:
        lab = cv2.cvtColor(person, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)
        lab[:,:,1:] -= 128.0
        lab[:,:,1:] /= 128.0
        shirt_hist = np.mean(lab[:int(0.6*h), :, 1:], axis=(0,1))
        pants_hist = np.mean(lab[int(0.6*h):, :, 1:], axis=(0,1))

    shirt_hist = cv2.normalize(shirt_hist, shirt_hist).flatten()
    pants_hist = cv2.normalize(pants_hist, pants_hist).flatten()
    return shirt_hist, pants_hist
```

---

### 2️⃣ `reid_tracker.py` update

Compute color distance separately, combine with geometry & motion.

```python
color_cost = 0.6 * D_shirt + 0.4 * D_pants
```

Normalize feature vector; keep existing EMA smoothing.

---

### 3️⃣ `fusion_logger.py` enhancement

Append average color per ID to CSV after each run:

```
id,frames,shirt_L,shirt_a,shirt_b,pants_L,pants_a,pants_b
```

(Or HSV means if using HSV.)

---

### 4️⃣ `visualizer_bbox.py`

Draw two color squares beside ID label using logged shirt/pants mean colors for visual verification.

---

## 📊 Evaluation Metrics

| Metric            | Target                                  | Notes                                   |
| ----------------- | --------------------------------------- | --------------------------------------- |
| ID retention      | ≥ 80 %                                  | Stable labels                           |
| Average ID switch | ≤ 3                                     | Smooth tracking                         |
| Color accuracy    | ≥ 85 % subjective match to real clothes | Visual inspection via overlay           |
| FPS               | ≥ 16                                    | Maintain real-time speed on Jetson Nano |

---

## 🚀 Expected Outcome

* Correct, vivid color visualization (no more gray bias).
* Accurate per-ID shirt/pants logging.
* Improved ID retention (~70–85 %).
* Stable base for final **Week 6 – Gait & Temporal Fusion**.

---
