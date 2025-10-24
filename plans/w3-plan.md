---

# 🧩 Week 3 – Re-ID + Approx Height & Distance Estimation

## 🎯 Objective

Extend the existing YOLOv8-Pose pipeline to perform **identity-consistent tracking** using Re-ID features (height, ratios) while estimating **relative distance and real-world height** from 2-D camera geometry.
This stage replaces the LiDAR fusion step with a **pseudo-depth model** based on body proportions and pixel geometry.

---

## 🧠 Key Idea

Use the person’s **torso–leg ratio** as a proxy for perspective scaling:
[
Z_{approx} = k_1 \times \frac{r_{tl,ref}}{r_{tl,measured}}
]
and
[
H_{real} = \frac{h_{px} \cdot Z_{approx}}{f}
]
where ( f ) is the focal length in pixels (from calibration), ( r_{tl,ref}≈0.53 ) is the average adult ratio.

These approximations let us compare physical-scale features and maintain consistent IDs over time.

---

## 🧩 Modules

### 1️⃣ `reid_tracker.py` (update)

Add geometric normalization + per-ID Re-ID logic.

```python
def match_tracks(detections, tracks):
    """Match detections to existing IDs using normalized height_m, shoulder/hip ratio, and small motion prior."""
```

* Compute `height_m_est` using ratio-based correction.
* Match using weighted cost:

  ```
  cost = 0.5*|h_m_det - h_m_track|
       + 0.4*|r_sh_det - r_sh_track|
       + 0.1*(distance of centers/img_diag)
  ```
* Use Hungarian assignment and gating (`cost < 0.35`).

---

### 2️⃣ `geometry_estimator.py` (new)

Handles calibration and approximate distance/height math.

```python
def calibrate_reference(ref_height_m=1.70, ref_distance_m=2.0, ref_height_px=360):
    """Compute scale constant k1 and focal length f_px."""
    f_px = (ref_height_px * ref_distance_m) / ref_height_m
    return f_px

def estimate_distance_ratio(torso_leg_ratio, ref_ratio=0.53, k1=1.0):
    """Return relative distance multiplier (closer => smaller value)."""
    return k1 * (ref_ratio / torso_leg_ratio)

def estimate_height_m(height_px, torso_leg_ratio, f_px, ref_ratio=0.53, k1=1.0):
    """Approximate real-world height (m)."""
    Z = estimate_distance_ratio(torso_leg_ratio, ref_ratio, k1)
    return (height_px * Z) / f_px
```

---

### 3️⃣ `fusion_logger.py`

Append new physical-space metrics:

```
frame,track_id,height_px,height_m_est,dist_est,shoulder_hip,torso_leg,conf
```

---

### 4️⃣ `visualizer_bbox.py`

Replace skeleton overlay with color-coded bounding boxes:

* Label: `ID:{id}  H:{h_m:.2f}m  D:{dist:.2f}`
* Colors persist per ID.

---

### 5️⃣ `main_reid_geometry.py`

Integrate all modules:

```python
for frame in video:
    detections = pose_extractor(model, frame)
    for det in detections:
        det["height_m"] = estimate_height_m(det["height_px"], det["torso_leg"], f_px)
    tracks = tracker.update(detections)
    log_fused_features(tracks)
    draw_bbox_labels(frame, tracks)
```

---

## 📊 Evaluation Metrics

| Metric                        | Target          | Meaning                               |
| ----------------------------- | --------------- | ------------------------------------- |
| ID Retention Rate             | ≥ 90 %          | Frames where the same person keeps ID |
| Height std per ID             | ≤ 0.15 m        | Scale stability                       |
| Relative Distance Consistency | ≤ 10 % variance | Z approx stability                    |
| FPS                           | ≥ 18 FPS (CPU)  | Real-time feasibility                 |

---

## 📦 Deliverables

* `outputs/fused_features_sim.csv` — includes height_m_est & dist_est
* `outputs/reid_overlay.mp4` — stable ID labels with height + distance
* `outputs/reid_summary.txt` — mean/std per ID and tracking stability
* Updated documentation: calibration constants + matching thresholds

---

## 🚀 Next Week Preview

Week 4 will add **gait features** and finalize **Re-ID fusion weights** before integrating real LiDAR.

---

