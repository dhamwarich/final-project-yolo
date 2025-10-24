

## ✅ Week 5.8 – Upper-Body Color Robustness Plan

### 🎯 Objective

Stabilize *shirt color* detection under pose and lighting variation by

1. isolating clothing pixels using person segmentation or pose, and
2. measuring color over a larger, robust region (not a small center patch).

---

### 🧩 Implementation Tasks

#### 1️⃣ Replace “center patch” with segmentation mask

Use YOLOv8-seg or lightweight **human segmentation** (e.g., `cv2.dnn.readNet('deeplabv3.onnx')`).

```python
mask = (seg_output == person_class_id)
shirt_mask = mask[int(0.25*h):int(0.55*h), :]    # upper torso
pants_mask = mask[int(0.65*h):, :]               # legs
```

→ Color only from true clothing pixels, not skin/background.

If segmentation isn’t available yet:

```python
hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
S_mask = S > 60
V_mask = (V > 40) & (V < 220)
mask = S_mask & V_mask
mean = cv2.mean(hsv, mask.astype(np.uint8))
```

This uses saturation/brightness gating to skip skin highlights.

---

#### 2️⃣ Expand shirt region vertically (20 % → 40 %)

Sampling window:
`shirt_y1=0.20*h, shirt_y2=0.60*h`
`shirt_x1=0.25*w, shirt_x2=0.75*w`
→ Wider zone averages folds/shadows out.

---

#### 3️⃣ Adaptive smoothing

If hue variance > 10° → increase temporal α from 0.3 → 0.5 for next frames (adaptive EMA).

---

#### 4️⃣ Re-weight color feature in Re-ID cost

Temporarily reduce color weight to 0.25 (shirt 0.15 + pants 0.10) until hue stabilizes.

---

#### 5️⃣ Add **confidence weighting** by saturation

```
color_conf = np.clip(S_mean/128, 0, 1)
cost_color *= (1 - 0.5*(1 - color_conf))
```

→ Low-saturation (white/gray) shirts won’t dominate matching.

---

#### 6️⃣ Logging additions

`reid_color_stability.csv`

```
track_id,shirt_H_mean,shirt_H_std,shirt_S_mean,shirt_conf,
pants_H_mean,pants_H_std,pants_S_mean,pants_conf
```

Target: σ ≤ 5°,  S_mean ≥ 70.

---

### 📊 Evaluation Metrics

| Metric             | Target | Notes              |
| ------------------ | ------ | ------------------ |
| Shirt hue variance | ≤ 5°   | stable color       |
| Pants hue variance | ≤ 5°   | keep current       |
| ID retention       | ≥ 75 % | improved stability |
| Avg FPS            | ≥ 15   | maintain real-time |

---

### 🚀 Expected Outcome

* Shirt hue variance ↓ from 18° → < 6°.
* Overlay patches consistent across frames.
* Re-ID IDs stop jumping (retention ≈ 80 %).
* Pants stability preserved.

---

### ✅ Deliverables

* `appearance_extractor.py` updated (segmentation + adaptive smoothing)
* `reid_color_stability.csv` + visual overlay `reid_overlay_final.mp4`
* Integrated report with σ and confidence columns

---