
# 🧩 Week 5 – Color-Region Re-ID Refinement

## 🎯 Objective

Achieve robust, low-switch identity tracking by fusing **separate upper- and lower-body color descriptors** with light-weight geometric and motion cues.
Remove unreliable torso-leg ratio and pseudo-height from all computations.

---

## 🧠 Overview

Previous results showed:

* Track count ≈ real people ✅
* Low ID-switches ✅
* But ID retention ≈ 42 % ⚠️ (mainly due to overlapping clothing)
* Torso/leg ratio unstable when frame is cropped ⚠️

This phase stabilizes identity matching through **color-region separation** and **re-balanced feature weighting**.

---

## 🧩 Feature Vector v5

| Category   | Feature                                 | Description             | Weight |
| ---------- | --------------------------------------- | ----------------------- | ------ |
| Geometry   | Shoulder / Hip ratio                    | Body build width ratio  | 0.25   |
| Geometry   | Bounding-box aspect ratio               | Posture & scale hint    | 0.15   |
| Appearance | **Shirt color histogram** (16 bins LAB) | Distinct upper clothing | 0.35   |
| Appearance | **Pants color histogram** (16 bins LAB) | Distinct lower clothing | 0.15   |
| Motion     | IoU + center distance                   | Temporal continuity     | 0.10   |

Weights sum = 1.00
→ Height ❌, Torso/Leg ❌ removed.

---

## ⚙️ Matching Cost

[
cost = 0.25ΔR_{sh} + 0.15ΔAR +
0.35D_{shirt} + 0.15D_{pants} +
0.10·motion
]
where

* (D) = Bhattacharyya distance between LAB histograms (8–16 bins)
* `motion` = 0.5 × (1 – IoU) + 0.5 × (center_dist / img_diag)
* Matching gate = `cost < 0.45`

---

## 🧩 Implementation Tasks

### 1️⃣ `appearance_extractor.py`

* Convert to LAB color space: `cv2.COLOR_BGR2LAB`
* Compute 16-bin histograms for upper 60 % (shirt) and lower 40 % (pants)
* Return dict with: `shirt_hist`, `pants_hist`, `sat_mean`

### 2️⃣ `reid_tracker.py`

* Drop torso/leg term
* Compute separate color distances and blend as 0.6 × shirt + 0.4 × pants
* Apply 5 frame EMA for ratios & color histograms
* Keep short-term ID memory (5 s window)

### 3️⃣ `fusion_logger.py`

```
frame,track_id,shoulder_hip,aspect,shirt_hist,pants_hist,conf
```

### 4️⃣ `visualizer_bbox.py`

* Bounding boxes color-coded per ID
* Add two small color patches (upper/lower) beside ID text for visual debugging

---

## 📊 Evaluation Metrics

| Metric            | Target           | Notes                    |
| ----------------- | ---------------- | ------------------------ |
| ID retention rate | ≥ 80 %           | Stable identity labels   |
| Avg ID switches   | ≤ 3 / 350 frames | Smooth tracking          |
| TL variance       | –                | Term removed (NA)        |
| FPS               | ≥ 16             | Real-time on Jetson Nano |

---

## 🚀 Expected Outcome

* Consistent IDs even with partial body view
* Color patches visually match real clothing
* Fewer duplicate IDs across same person
* Stable performance (≈ 16–18 FPS)

---

## 🔭 Next Step Preview (Week 6)

Integrate **gait signature / temporal embedding** to finalize the full Re-ID model before LiDAR fusion phase.

---
