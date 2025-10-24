

# 🧩 Week 4.5 – Re-ID Tuning & Memory Stabilization (No Height)

## 🎯 Objective

Improve ID stability using **appearance + geometry** only, removing pseudo-height.
Introduce **short-term memory** and **histogram-based color features** for robust Re-ID.

---

## 🧠 Feature set (simplified, stable)

| Category   | Feature                        | Description                        |
| ---------- | ------------------------------ | ---------------------------------- |
| Geometry   | Shoulder–Hip ratio             | Distinguishes body proportion      |
| Geometry   | Torso–Leg ratio                | Perspective consistency (smoothed) |
| Geometry   | Bbox aspect ratio              | Rough build / posture              |
| Appearance | Top Hue histogram (16 bins)    | Distinguishes clothing color       |
| Appearance | Bottom Hue histogram (16 bins) | Pants/skirt tone                   |
| Appearance | Texture variance               | Pattern richness                   |
| Motion     | IoU + center proximity         | Frame-to-frame continuity          |

No absolute or metric heights — only relative geometry.

---

## ⚙️ Matching cost

[
cost = 0.25ΔR_{sh} + 0.15ΔR_{tl} + 0.10ΔAR +
0.35·color_distance + 0.05Δtexture + 0.10·motion
]

* **Color distance** = Bhattacharyya between 16-bin histograms.
* **Motion** = 0.5·(1–IoU) + 0.5·(center_dist/img_diag).
* Gating threshold: `cost < 0.45`.

---

## 🧩 New Modules / Updates

### 1️⃣ `appearance_extractor.py`

Expand to extract top/bottom 16-bin histograms instead of single HSV mean.

### 2️⃣ `reid_tracker.py`

* Remove all `height_m` terms.
* Add `track_memory` dict to store last 5 s of lost IDs:

  ```python
  if cost_to_lost < 0.25 → reuse old ID
  ```
* Keep EMA on ratios & color histograms.

### 3️⃣ `filter_smooth.py`

Temporal smoothing for ratios:

```python
r_tl = α*r_tl_curr + (1-α)*r_tl_prev  # α=0.3
```

### 4️⃣ `fusion_logger.py`

Simplify log:

```
frame,track_id,r_sh,r_tl,aspect,top_hist,bottom_hist,texture,conf
```

### 5️⃣ `visualizer_bbox.py`

Use consistent colored boxes per ID + hue patch (top clothing sample).

---

## 📊 Evaluation Metrics

| Metric          | Target             | Notes                 |
| --------------- | ------------------ | --------------------- |
| ID retention    | ≥ 75 %             | stable labeling       |
| Avg ID switches | ≤ 3 per 350 frames |                       |
| FPS             | ≥ 16 FPS           | maintain speed        |
| TL variance     | ≤ 10 %             | geometric consistency |

---

## 🚀 Expected Results

* Track count ≈ actual people (6–7 IDs max)
* No height spikes or metric drift
* Re-ID relies on consistent color + body shape
* Foundation ready for LiDAR distance once available

---