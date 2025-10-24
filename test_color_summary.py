"""
Test the updated color summary with colors in text file.
"""

import csv
import os
from tracking.fusion_logger_v4_crop import summarize_colorized_reid

# Create test data
os.makedirs("outputs", exist_ok=True)

# Create mock tracking CSV
tracking_csv = "outputs/reid_colorized_test.csv"
with open(tracking_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track_id', 'shoulder_hip', 'aspect', 'shirt_hist', 'pants_hist', 'x_center', 'y_center', 'conf'])
    
    # Track 0 - appears in 50 frames
    for i in range(50):
        writer.writerow([i, 0, 1.25, 2.1, '0.1 0.2 0.3 0.2 0.1 0 0 0 0 0 0 0 0 0 0 0', '0 0 0.1 0.2 0.3 0.2 0.1 0 0 0 0 0 0 0 0 0', 500, 500, 0.9])
    
    # Track 1 - appears in 30 frames
    for i in range(30):
        writer.writerow([i, 1, 1.30, 2.2, '0 0 0 0 0 0 0.3 0.3 0.2 0.1 0.1 0 0 0 0 0', '0 0 0 0 0.1 0.2 0.3 0.2 0.1 0 0 0 0 0 0 0', 1000, 500, 0.85])
    
    # Track 2 - appears in 20 frames
    for i in range(20):
        writer.writerow([i, 2, 1.35, 2.3, '0 0 0 0 0 0 0 0 0 0.3 0.3 0.2 0.1 0.1 0 0', '0.2 0.3 0.2 0.1 0.1 0 0 0 0 0 0 0 0 0 0 0', 1500, 500, 0.88])

# Create mock color CSV
color_csv = "outputs/reid_colorized_test_color_summary.csv"
with open(color_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['track_id', 'frames', 'shirt_H', 'shirt_S', 'shirt_V', 'shirt_color_name', 'shirt_full_desc', 'pants_H', 'pants_S', 'pants_V', 'pants_color_name', 'pants_full_desc'])
    writer.writerow([0, 50, 110.5, 180.2, 200.1, 'blue', 'blue', 30.2, 150.5, 180.3, 'yellow', 'yellow'])
    writer.writerow([1, 30, 60.8, 170.9, 190.2, 'green', 'green', 45.3, 120.4, 170.8, 'green', 'green'])
    writer.writerow([2, 20, 5.2, 200.1, 185.5, 'red/pink', 'red/pink', 15.8, 90.2, 160.4, 'orange/brown', 'pale orange/brown'])

print("="*75)
print("TESTING COLOR SUMMARY WITH TRACK COLORS")
print("="*75)

print("\n1. Created test tracking CSV with 3 tracks")
print("   - Track 0: 50 frames")
print("   - Track 1: 30 frames")
print("   - Track 2: 20 frames")

print("\n2. Created test color CSV with color names")
print("   - Track 0: blue shirt, yellow pants")
print("   - Track 1: green shirt, green pants")
print("   - Track 2: red/pink shirt, pale orange/brown pants")

print("\n3. Generating summary with colors...")
summarize_colorized_reid(tracking_csv)

print("\n✅ Test complete! Check outputs/reid_summary_colorized_test.txt")
print("\nExpected output for each track:")
print("  Track ID X:")
print("    Appearances: XX frames (XX.X%)")
print("    Shoulder/Hip: X.XX ± X.XX")
print("    Aspect ratio: X.XX")
print("    Shirt color: [color name] (H=XXX.X°, S=XXX.X)")
print("    Pants color: [color name] (H=XXX.X°, S=XXX.X)")
print("    ID retention: ✓ (target: ≥75%)")
