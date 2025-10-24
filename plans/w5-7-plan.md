

# 🧩 Week 5.7 – Color Stability Refinement

## 🎯 Objective

Stabilize per-frame color detection by sampling **smaller, fixed-position zones** inside each bounding box (rather than the whole upper/lower region).
Reduce frame-to-frame hue variance and remove flicker from overlay patches.

---

## 🧠 Problem Summary

* Detected colors fluctuate each frame due to box jitter & illumination.
* Current top/bottom crops (60 / 40 %) still include edges, sleeves, and background.
* Need temporally smoothed, center-focused color sampling and box stabilization.

---

## 🧩 Implementation Tasks

### 1️⃣ `appearance_extractor.py` — Centralized Color Sampling

Replace full-region mean with a small center patch (≈ 30 % of width × 20 % of height).

```python
import cv2, numpy as np

def extract_center_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    person = frame[y1:y2, x1:x2]

    # --- define fixed shirt & pants windows (center only) ---
    shirt_y1 = int(0.25*h); shirt_y2 = int(0.45*h)
    pants_y1 = int(0.70*h); pants_y2 = int(0.90*h)
    shirt_x1 = int(0.35*w); shirt_x2 = int(0.65*w)

    shirt = person[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    pants = person[pants_y1:pants_y2, shirt_x1:shirt_x2]

    def mean_hsv(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H,S,V = cv2.split(hsv)
        mask = S > 50
        mean = cv2.mean(hsv, mask=mask.astype(np.uint8))
        return mean[:3]

    return mean_hsv(shirt), mean_hsv(pants)
```

→ This keeps a **stable patch at the torso and thigh center**, independent of pose or small box drift.

---

### 2️⃣ Add Bounding Box Smoothing

Before color extraction, apply exponential moving average on box coords:

```python
beta = 0.6
bbox_smooth = beta*np.array(curr_bbox) + (1-beta)*np.array(prev_bbox)
```

→ Reduces jitter by ~70 %.

---

### 3️⃣ Temporal Color Smoothing

Smooth hue over 5 frames:

```python
alpha = 0.3
H_smooth = alpha*H_curr + (1-alpha)*H_prev
```

Maintain per-track rolling hue buffer (`deque(maxlen=5)`).

---

### 4️⃣ Color Variance Logger

Add to summary:

```
shirt_hue_std, pants_hue_std
```

Target ≤ 5 °.
Helps quantify flicker reduction.

---

### 5️⃣ `visualizer_bbox.py` — Stable Color Patches

Use smoothed HSV values for overlay; same vivid conversion:

```python
color_shirt = hsv_to_bgr(int(H_smooth),200,200)
color_pants = hsv_to_bgr(int(Hp_smooth),200,200)
```

---

### 6️⃣ `reid_summary_with_colors.csv` — Extended Fields

```
track_id,frames,shirt_color_name,pants_color_name,
shirt_H_mean,shirt_H_std,pants_H_mean,pants_H_std
```

---

## 📊 Evaluation Metrics

| Metric                  | Target  | Notes             |
| ----------------------- | ------- | ----------------- |
| Hue variance (σ per ID) | ≤ 5 °   | stable colors     |
| ID retention            | ≥ 75 %  | maintain tracking |
| Visual color flicker    | minimal | overlay steady    |
| FPS                     | ≥ 16    | real-time         |

---

## 🚀 Expected Outcome

* Shirt/pants colors stay constant throughout the sequence.
* Overlay patches no longer flicker.
* Hue variance per track ≤ 5 °, ID retention ≈ 80 %.
* Logs include mean & std for color validation.

---

✅ **Deliverables**

* `appearance_extractor.py` (updated center sampling)
* `bbox_smoothing` utility (EMA)
* `temporal_color_smoothing` buffer
* `reid_summary_with_colors.csv` (with hue std fields)
* `reid_overlay_stable.mp4` (stable color visualization)

--