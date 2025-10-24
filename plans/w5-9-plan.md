
---

## 🎯 Week 5.9 — Temporal Re-ID & Feature Memory Refinement

### 🔍 Insight

From the last four experiments:

* **Color features** have low confidence (S≈0.3), huge σ≈47° → unreliable.
* **Geometric ratios (shoulder-hip, aspect)** are relatively consistent (σ<0.1).
* **Temporal continuity** is strong — motion between adjacent frames is smooth except when occluded.

So instead of trying to “fix” color, the better strategy is to:

> **Stop trusting instantaneous color. Trust temporal coherence.**

---

## 🧩 Plan Summary

We’ll **de-emphasize per-frame color** and introduce a **temporal identity memory module** that uses motion and geometry to stabilize IDs across frames, then re-inject averaged color as a low-weight contextual cue.

---

### 1️⃣ Feature Memory Bank (Short-Term)

Maintain a rolling memory `M_id` for each track:

```python
M_id = {
  'geom': EMA(last_geom, α=0.3),
  'motion': last_center_velocity,
  'color': EMA(last_color, α=0.1),
  'age': frames_since_seen
}
```

When a new detection appears, compute:

```
S = 0.6*geom_sim + 0.3*motion_sim + 0.1*color_sim
```

→ IDs now depend mostly on *shape & motion*, not hue.

---

### 2️⃣ Motion Similarity

Use centroid displacement:

```python
motion_sim = exp(-||Δv|| / τ)
```

where `Δv` = difference in velocity vectors over last 2 frames, τ≈0.2 m/s.
This helps keep the same person ID through motion blur.

---

### 3️⃣ Color Usage Change

* Keep color logging for analytics but **drop its weight to 0.1 total**.
* If `color_conf < 0.4`, ignore color in matching completely.
* For overlay visualization, show last stable average color (EMA over 15 frames) instead of per-frame.

---

### 4️⃣ Temporal Occlusion Handling

Introduce **invisibility buffer**:

```
if missed < 10 frames → keep last state, decay confidence
else → mark as lost
```

This prevents ID switches when someone passes behind another shopper for ~0.3 s.

---

### 5️⃣ Logging Enhancements

Add to `reid_temporal_summary.csv`:

```
track_id,appearances,ID_switches,avg_geom_std,avg_motion_std,color_conf_mean
```

and to overlay:

* Grey out label if confidence < 0.3.
* Display “ID retained X frames” metric.

---

### 6️⃣ Evaluation Metrics

| Metric           | Target           | Notes                        |
| ---------------- | ---------------- | ---------------------------- |
| ID retention     | ≥ 85 %           | sustained through occlusion  |
| Avg ID switches  | ≤ 2 / 350 frames | smoother tracking            |
| Color confidence | logged only      | no longer affects score      |
| FPS              | ≥ 16             | maintain Jetson-viable speed |

---

### 7️⃣ Expected Outcome

* Smooth, stable tracking even if color flickers.
* Identity persistence driven by geometry + motion.
* Overlay shows consistent colors (from long EMA) for aesthetics only.
* Color CSV still available for research but ignored in logic.

---

✅ **Deliverables**

* `reid_tracker.py` updated (feature memory & motion fusion)
* `temporal_reid.py` module (memory, decay, motion sim)
* `reid_temporal_summary.csv`
* `reid_overlay_temporal.mp4` (stable IDs, minimal flicker)

-