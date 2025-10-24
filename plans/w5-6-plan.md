
# 🧩 Week 5.6 – Color Logging & Visualization Fixes

## 🎯 Objective

Ensure per-person clothing colors are extracted and visualized accurately.
Fix the “gray overlay” bug by cropping tighter, masking low-saturation pixels before averaging, converting HSV to vivid RGB, and logging readable color names directly tied to each track ID.

---

## 🧠 Overview

**Current issues**

* Gray overlay patches even on vivid shirts.
* Color CSV not linked to track IDs.
* RGB means clustered 100–140 → low-contrast gray.

**Goal**

1. Extract shirt/pants colors from real clothing pixels (not background).
2. Mask out gray/low-saturation areas.
3. Map hue to simple color names (pink, blue, green …).
4. Merge results with main Re-ID summary.
5. Show vivid, correct patches in overlay.

---

## 🧩 Implementation Tasks

### 1️⃣ `appearance_extractor.py` — Tight Crop + Masked Color Averaging

```python
import cv2, numpy as np

def extract_color_regions(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    pad = 0.1
    x1 += int((x2-x1)*pad); x2 -= int((x2-x1)*pad)
    y1 += int((y2-y1)*pad); y2 -= int((y2-y1)*pad)
    person = frame[y1:y2, x1:x2]
    h = person.shape[0]

    top = person[:int(0.6*h), :]
    bottom = person[int(0.6*h):, :]

    def mean_hsv(region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        mask = S > 50
        mean = cv2.mean(hsv, mask=mask.astype(np.uint8))
        return mean[:3]  # H,S,V

    shirt_HSV = mean_hsv(top)
    pants_HSV = mean_hsv(bottom)
    return shirt_HSV, pants_HSV
```

---

### 2️⃣ `color_utils.py` — Color Conversion & Naming

```python
import cv2, numpy as np

def hsv_to_bgr(h, s=200, v=200):
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return tuple(int(x) for x in bgr)

def hue_to_name(h):
    if h < 10 or h >= 170: return "red/pink"
    elif h < 25: return "orange/brown"
    elif h < 35: return "yellow"
    elif h < 85: return "green"
    elif h < 125: return "blue"
    elif h < 145: return "violet/purple"
    elif h < 160: return "magenta"
    else: return "unknown"
```

---

### 3️⃣ `reid_tracker.py` — Integrate Colors with Tracks

```python
shirt_HSV, pants_HSV = extract_color_regions(frame, bbox)
track['shirt_HSV'] = shirt_HSV
track['pants_HSV'] = pants_HSV
```

* After all frames, group by `track_id` and average H,S,V.
* Apply `hue_to_name()` for readable shirt/pants color fields.
* Merge these into `reid_summary_with_colors.csv`.

---

### 4️⃣ `fusion_logger.py` — Log Per-ID Colors

```
track_id,frames,
shirt_H,shirt_S,shirt_V,shirt_color_name,
pants_H,pants_S,pants_V,pants_color_name
```

* Write after grouping (by ID) to ensure one line per person.
* Merge with your main Re-ID summary.

---

### 5️⃣ `visualizer_bbox.py` — Display Vivid Color Patches

```python
from color_utils import hsv_to_bgr

color_shirt = hsv_to_bgr(shirt_H)
color_pants = hsv_to_bgr(pants_H)
cv2.rectangle(frame, (x1, y1-20), (x1+20, y1), color_shirt, -1)
cv2.rectangle(frame, (x1+25, y1-20), (x1+45, y1), color_pants, -1)
```

---

### 6️⃣ `postprocess_merge.py` — Merge Summaries

```python
import pandas as pd
reid = pd.read_csv("outputs/reid_summary_normalized.txt", sep='\t', engine='python')
colors = pd.read_csv("outputs/reid_color_summary_by_id.csv")
merged = pd.merge(reid, colors, on='track_id', how='left')
merged.to_csv("outputs/reid_summary_with_colors.csv", index=False)
```

---

## 📊 Evaluation Metrics

| Metric                   | Target          | Notes                                 |
| ------------------------ | --------------- | ------------------------------------- |
| **Color match accuracy** | ≥ 90 %          | Visual check vs. video                |
| **Color patch clarity**  | vivid, non-gray | overlay should reflect actual clothes |
| **ID retention**         | ≥ 75 %          | should remain stable                  |
| **Average FPS**          | ≥ 16            | real-time speed                       |

---

## 🧩 Expected Outputs

1. `reid_summary_with_colors.csv` — main summary + shirt/pants names
2. `reid_color_summary_by_id.csv` — numeric H,S,V averages per ID
3. `reid_overlay_colorized.mp4` — overlay with vivid color patches
4. `reid_logs.txt` — debug lines showing per-ID color names and confidence

---

## 🚀 Expected Results

* Overlays display true vivid clothing hues (no gray).
* Each ID logs its shirt/pants color and human-readable names.
* Color CSV and Re-ID summary fully synchronized by ID.
* Ready foundation for Week 6 (adding gait features + temporal fusion).

---

✅ **Deliverables**

* `appearance_extractor.py` updated (HSV masking + tight crop)
* `color_utils.py` added (hue→name mapping)
* Color logs merged with Re-ID summary
* Overlay color patch validated
