# Week 2.5: Feature Quality Refinement

## 🎯 Objective

Clean and stabilize per-frame geometric data through outlier filtering, temporal smoothing, and improved visualization for accurate Re-ID training.

## 🆕 Enhancements

### 1. **Outlier Filtering**
Rejects frames with implausible pose measurements to ensure data quality.

**Validation Thresholds:**
- `height_px > 100` - Minimum realistic height
- `0.5 < shoulder_hip < 3.0` - Realistic shoulder/hip ratio range
- `0.2 < torso_leg < 3.0` - Realistic torso/leg ratio range
- No NaN or infinity values

### 2. **Temporal Smoothing (EMA)**
Reduces frame-to-frame flicker using Exponential Moving Average.

**Formula:** `smoothed = α × current + (1 - α) × previous`

**Default α = 0.2:**
- 20% weight on current frame
- 80% weight on previous (smoothed) value
- Lower α = more smoothing, less responsiveness

### 3. **Per-Person Text Overlay**
Displays metrics above each detected person instead of fixed top-left position.

**Format:** `H:320 SH:1.52 TL:0.83`

### 4. **Summary Statistics Report**
Auto-generates statistical analysis after processing.

**Metrics:**
- Mean, standard deviation, outlier percentage for each feature
- Target evaluation (variance < 0.1, outliers < 10%)
- Saved to `outputs/summary.txt`

## 📁 New/Updated Files

```
features/
├── filter_smooth.py         # NEW: Filtering + EMA logic
├── visualize_pose.py        # UPDATED: Per-person overlay
└── feature_logger.py        # UPDATED: Summary generation

main_features_refined.py     # NEW: Refined integration
test_refined.py              # NEW: Test script
```

## 🔧 Module Details

### `filter_smooth.py`

#### `is_valid_frame(features: dict) -> bool`
Validates feature measurements against thresholds.

**Returns:** `True` if valid, `False` if outlier

**Example:**
```python
feats = {'height_px': 50, 'shoulder_hip': 4.0, 'torso_leg': 0.8}
is_valid_frame(feats)  # False (height too low, SH ratio too high)
```

#### `smooth_features(curr: dict, prev: dict, alpha: float = 0.2) -> dict`
Applies exponential moving average to reduce noise.

**Example:**
```python
curr = {'height_px': 320, 'shoulder_hip': 1.5, 'torso_leg': 0.8}
prev = {'height_px': 310, 'shoulder_hip': 1.48, 'torso_leg': 0.82}
smoothed = smooth_features(curr, prev, alpha=0.2)
# Result: {'height_px': 312.0, ...}  # Smoothed between curr and prev
```

### `visualize_pose.py` (Updated)

#### `overlay_metrics(frame, features, bbox_center=None)`
Now accepts `bbox_center` parameter for per-person placement.

**Example:**
```python
# Old: Always at top-left
overlay_metrics(frame, feats)

# New: Above person's bounding box
overlay_metrics(frame, feats, feats['bbox_center'])
```

### `feature_logger.py` (Updated)

#### `summarize_log(csv_path: str)`
Generates comprehensive statistics report.

**Output example (`outputs/summary.txt`):**
```
============================================================
FEATURE SUMMARY REPORT
============================================================

Total frames processed: 872
Average confidence: 0.89

------------------------------------------------------------
METRIC STATISTICS
------------------------------------------------------------

Height (px):
  Mean:     410.20
  Std:      38.70
  Outliers: 5.2%

Shoulder/Hip Ratio:
  Mean:     1.31
  Std:      0.14
  Outliers: 3.8%

Torso/Leg Ratio:
  Mean:     0.79
  Std:      0.09
  Outliers: 4.1%

============================================================
TARGET EVALUATION
============================================================

Overall outlier rate: 4.4% ✓ (Target: <10%)
Shoulder/Hip variance: 0.14 ✗ (Target: <0.1)
Torso/Leg variance: 0.09 ✓ (Target: <0.1)

============================================================
```

## 🚀 Usage

### Run Refined Pipeline

```bash
python main_features_refined.py
```

**What it does:**
1. Loads YOLOv8n-pose model
2. Processes video with filtering and smoothing
3. Shows real-time visualization with per-person overlays
4. Saves refined features to `outputs/features_refined.csv`
5. Saves annotated video to `outputs/refined_overlay.mp4`
6. Generates `outputs/summary.txt` with statistics

### Quick Test

```bash
python test_refined.py
```

Tests filtering and smoothing on single frame.

## 📊 Evaluation Targets

| Metric              | Target            | Purpose                    |
| ------------------- | ----------------- | -------------------------- |
| Outlier rate        | < 10% of frames   | Data quality               |
| Ratio variance      | Std < 0.1         | Consistency for Re-ID      |
| FPS                 | ≥ 18 FPS          | Real-time capability       |
| Overlay readability | One label/person  | Debugging clarity          |

## 🔍 Key Parameters

### Smoothing Alpha (`SMOOTHING_ALPHA`)

**Default: 0.2**

Adjust in `main_features_refined.py`:
```python
SMOOTHING_ALPHA = 0.2  # Range: 0.0 - 1.0
```

**Effects:**
- **α = 0.0:** Maximum smoothing (static, uses only first frame)
- **α = 0.2:** Recommended (balanced smoothing)
- **α = 0.5:** Moderate smoothing
- **α = 1.0:** No smoothing (raw values)

**Use cases:**
- Lower α (0.1-0.2): Noisy detections, stable subjects
- Higher α (0.4-0.6): Fast movements, high FPS

### Validation Thresholds

Adjust in `features/filter_smooth.py`:
```python
# In is_valid_frame()
if height_px < 100:  # Minimum height threshold
if shoulder_hip < 0.5 or shoulder_hip > 3.0:  # SH ratio range
if torso_leg < 0.2 or torso_leg > 3.0:  # TL ratio range
```

## 📈 Expected Improvements Over Week 2

| Aspect              | Week 2 (Basic)     | Week 2.5 (Refined)      |
| ------------------- | ------------------ | ----------------------- |
| Invalid frames      | Logged (noise)     | Filtered out            |
| Frame-to-frame jitter| High               | Reduced by EMA          |
| Metric overlay      | Fixed top-left     | Per-person placement    |
| Quality insights    | Manual CSV review  | Auto summary report     |
| Data reliability    | Unknown            | Validated & quantified  |

## 🧪 Test Results

```
✓ Model loading functional
✓ Video processing working
✓ Detected 4 persons in test frame
✓ Outlier filtering active (invalid frames rejected)
✓ Temporal smoothing functional
✓ Per-person overlay positioning
✓ CSV logging with refined data
```

**Example filtered frame:**
- Height: 814.69px
- SH ratio: 1.86
- **TL ratio: 10.83** ← **REJECTED** (exceeds threshold of 3.0)

This indicates a partial or occluded detection correctly filtered out.

## 📝 Deliverables

- ✅ `features/filter_smooth.py` - Filtering and smoothing logic
- ✅ `main_features_refined.py` - Refined pipeline
- ✅ `outputs/features_refined.csv` - Cleaned feature log
- ✅ `outputs/summary.txt` - Statistical analysis
- ✅ `outputs/refined_overlay.mp4` - Improved visualization
- ✅ Documentation on thresholds and smoothing parameters

## 🔄 Pipeline Comparison

### Basic Pipeline (`main_features.py`)
```
Frame → Detect → Extract → Log → Visualize (fixed position)
```

### Refined Pipeline (`main_features_refined.py`)
```
Frame → Detect → Extract → Validate → Smooth → Log → Visualize (per-person)
                              ↓                             ↓
                          Filter Out                  Dynamic Position
                          Outliers
```

## 💡 Tips

1. **Adjust α based on motion:** Lower for static scenes, higher for dynamic
2. **Review summary.txt:** Check if targets are met before Re-ID training
3. **Tune thresholds:** If too many valid frames filtered, relax constraints
4. **Monitor invalid %:** Should be < 10% for good video quality
5. **Compare outputs:** Run both pipelines to see smoothing effects

## 🎯 Next Steps

1. Run refined pipeline on full video dataset
2. Review `summary.txt` for data quality metrics
3. Adjust thresholds if needed based on outlier rates
4. Use `features_refined.csv` for Re-ID model training (Week 3)
5. Verify smoothed features improve tracking consistency

---

**Status:** ✅ Week 2.5 refinements implemented and tested
**Ready:** Run `python main_features_refined.py` for production use
