# Archived Files

This directory contains deprecated and experimental code from the Re-ID tracking project development.

## Directory Structure

### `main_scripts/`
Old main pipeline scripts from previous weeks:
- `main_features.py` - Week 1: Basic feature extraction
- `main_features_refined.py` - Week 2: Refined features
- `main_reid_appearance.py` - Week 3: Appearance-based Re-ID
- `main_reid_colorized.py` - Week 4: Color-based Re-ID
- `main_reid_geometry.py` - Week 5.1: Geometry-based Re-ID
- `main_reid_normalized.py` - Week 5.2: Normalized features
- `main_reid_refined.py` - Week 5.3: Refined matching
- `main_reid_stable.py` - Week 5.5: Stable color extraction
- `main_reid_tuned.py` - Week 5.6: Tuned parameters

**Current active:** `main_reid_temporal.py` (Week 5.9), `main_reid_robust.py` (Week 5.8)

---

### `test_scripts/`
Old test scripts for development:
- `test_appearance.py` - Appearance feature tests
- `test_color_region.py` - Color region extraction tests
- `test_color_summary.py` - Color summary tests
- `test_colorized.py` - Colorized tracking tests
- `test_normalized.py` - Normalized feature tests
- `test_pipeline.py` - Pipeline integration tests
- `test_refined.py` - Refined tracking tests
- `test_reid.py` - Re-ID baseline tests
- `test_stable.py` - Stable color tests
- `test_tuned.py` - Tuned parameter tests

**Current active:** `test_temporal.py` (Week 5.9), `test_robust.py` (Week 5.8)

---

### `tracking_old_versions/`
Deprecated tracking module versions:

#### Appearance Extractors (v1-v3)
- `appearance_extractor.py` - v1: Basic color histograms
- `appearance_extractor_v2.py` - v2: Added HSV extraction
- `appearance_extractor_v3.py` - v3: Improved regions
- `appearance_extractor_v3_crop.py` - v3 crop: Cropped regions experiment
- `appearance_extractor_v3_norm.py` - v3 norm: Normalized colors
- `appearance_extractor_v3_stable.py` - v3 stable: Stable extraction

**Current active:** `appearance_extractor_v3_robust.py` (Week 5.8 - still used)

#### Re-ID Trackers (v1-v4)
- `reid_tracker.py` - v1: Basic Hungarian matching
- `reid_tracker_v2.py` - v2: Added memory
- `reid_tracker_v3.py` - v3: Improved cost matrix
- `reid_tracker_v4.py` - v4: Temporal smoothing
- `reid_tracker_v4_crop.py` - v4 crop: Cropped features
- `reid_tracker_v4_norm.py` - v4 norm: Normalized matching
- `reid_tracker_v4_stable.py` - v4 stable: Stable tracking

**Current active:** `reid_tracker_v5_temporal.py` (Week 5.9), `reid_tracker_v4_robust.py` (Week 5.8)

#### Fusion Loggers (v1-v4)
- `fusion_logger.py` - v1: Basic CSV logging
- `fusion_logger_v2.py` - v2: Added color logging
- `fusion_logger_v3.py` - v3: Per-ID summaries
- `fusion_logger_v4.py` - v4: Enhanced metrics
- `fusion_logger_v4_crop.py` - v4 crop: Cropped region logs
- `fusion_logger_v4_norm.py` - v4 norm: Normalized logs
- `fusion_logger_v4_stable.py` - v4 stable: Stability metrics

**Current active:** `fusion_logger_v5_temporal.py` (Week 5.9), `fusion_logger_v4_robust.py` (Week 5.8)

#### Visualizers (v1-v4)
- `visualizer_bbox.py` - v1: Basic bboxes
- `visualizer_bbox_v2.py` - v2: Color patches
- `visualizer_bbox_v3.py` - v3: Simplified
- `visualizer_bbox_v4.py` - v4: Enhanced display
- `visualizer_bbox_v4_crop.py` - v4 crop: Cropped viz
- `visualizer_bbox_v4_norm.py` - v4 norm: Normalized viz
- `visualizer_bbox_v4_stable.py` - v4 stable: Stable colors

**Current active:** `visualizer_bbox_v5_temporal.py` (Week 5.9), `visualizer_bbox_v4_robust.py` (Week 5.8)

#### Other
- `geometry_estimator.py` - Early geometry experiments

---

### `experiments/`
Experimental/demo files:
- `detect_yolo.py` - Basic YOLO detection demo
- `bus.jpg` - Test image
- `test.jpeg` - Test image

---

## Current Active Files (NOT in archive)

### Main Scripts
- `main_reid_temporal.py` - **Week 5.9**: Motion-dominant temporal tracking
- `main_reid_robust.py` - **Week 5.8**: Robust color with expanded regions

### Test Scripts
- `test_temporal.py` - **Week 5.9**: Temporal tracking tests
- `test_robust.py` - **Week 5.8**: Robust color tests

### Tracking Modules
- `temporal_reid.py` - **Week 5.9**: Feature memory & motion similarity
- `reid_tracker_v5_temporal.py` - **Week 5.9**: Motion-dominant tracker
- `fusion_logger_v5_temporal.py` - **Week 5.9**: Temporal metrics logger
- `visualizer_bbox_v5_temporal.py` - **Week 5.9**: Stable color visualizer
- `appearance_extractor_v3_robust.py` - **Week 5.8**: Robust color extractor (still used)
- `reid_tracker_v4_robust.py` - **Week 5.8**: Robust tracker (backup)
- `fusion_logger_v4_robust.py` - **Week 5.8**: Robust logger (backup)
- `visualizer_bbox_v4_robust.py` - **Week 5.8**: Robust visualizer (backup)

### Core Utilities (Always Active)
- `features/` - Pose extraction, ratios, filtering
- `utils/` - Bbox smoothing, color utilities

---

## Evolution Timeline

1. **Week 1-2**: Basic feature extraction
2. **Week 3**: Appearance-based Re-ID with color histograms
3. **Week 4**: Colorized tracking improvements
4. **Week 5.1**: Geometry-based matching
5. **Week 5.2**: Normalized features
6. **Week 5.3**: Refined matching cost matrix
7. **Week 5.5**: Stable color extraction
8. **Week 5.6**: Tuned parameters
9. **Week 5.7**: Multi-feature fusion (geometry + color + motion)
10. **Week 5.8**: Robust color with expanded regions & confidence weighting
11. **Week 5.9**: **Temporal tracking with motion-dominant matching** ← **Current**

---

## Why These Were Archived

- **Superseded**: Newer versions with better performance
- **Experimental**: Tested approaches that didn't work well
- **Redundant**: Variations that were merged into later versions
- **Low confidence**: Color-heavy approaches proven unreliable (σ≈47°)

---

## Recovery

If you need to recover any archived file:
```bash
cp archived/<subdir>/<filename> .
```

Or to compare with current version:
```bash
diff archived/<subdir>/<old_file> <new_file>
```
