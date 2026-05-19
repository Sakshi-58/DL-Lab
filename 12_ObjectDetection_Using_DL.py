# 12. Design an object detection model using deep neural networks for simple objects. 
# a.  
# Select appropriate dataset and perform data pre-processing  
# b.  Define architecture in terms of layers  
# c.  
# Evaluate Model performance Label the object with appropriate text. 




# !pip install ultralytics -q

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

print("All libraries imported successfully!")


# Load YOLOv8 nano model (pretrained on COCO dataset - 80 object classes)
model = YOLO('yolov8n.pt')

# Print model architecture/summary
print("=" * 60)
print("        YOLOv8 ARCHITECTURE SUMMARY")
print("=" * 60)
print("""
BACKBONE (Feature Extraction):
  Layer 1  : Conv2D(3→16,   3x3, stride=2)  + BN + SiLU
  Layer 2  : Conv2D(16→32,  3x3, stride=2)  + BN + SiLU
  Layer 3  : C2f Block (32  channels, 1 bottleneck)
  Layer 4  : Conv2D(32→64,  3x3, stride=2)  + BN + SiLU
  Layer 5  : C2f Block (64  channels, 2 bottlenecks)
  Layer 6  : Conv2D(64→128, 3x3, stride=2)  + BN + SiLU
  Layer 7  : C2f Block (128 channels, 2 bottlenecks)
  Layer 8  : Conv2D(128→256,3x3, stride=2)  + BN + SiLU
  Layer 9  : C2f Block (256 channels, 1 bottleneck)
  Layer 10 : SPPF (Spatial Pyramid Pooling Fast, 5x5)

NECK (Feature Pyramid Network):
  Layer 11 : Upsample (2x) + Concat with Layer 7
  Layer 12 : C2f Block (128 channels)
  Layer 13 : Upsample (2x) + Concat with Layer 5
  Layer 14 : C2f Block (64  channels)
  Layer 15 : Conv2D(64→64,  3x3, stride=2) + Concat with Layer 12
  Layer 16 : C2f Block (128 channels)
  Layer 17 : Conv2D(128→128,3x3, stride=2) + Concat with Layer 10
  Layer 18 : C2f Block (256 channels)

HEAD (Detection):
  Layer 19 : Detect Head on small   objects (80x80 grid)
  Layer 20 : Detect Head on medium  objects (40x40 grid)
  Layer 21 : Detect Head on large   objects (20x20 grid)
  Output   : [x, y, w, h, confidence, 80 class scores]
""")

print("Dataset : COCO (Common Objects in Context)")
print("Classes : 80 object categories")
print("Input   : 640 x 640 x 3 (RGB image)")
print("=" * 60)

# Show model info
model.info()


# All 80 COCO classes the model can detect
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

print("COCO Dataset Information")
print("=" * 40)
print(f"Total Classes     : {len(COCO_CLASSES)}")
print(f"Training Images   : ~118,000")
print(f"Validation Images : ~5,000")
print(f"Total Annotations : ~1.5 million")
print("=" * 40)
print("\nAll Detectable Classes:")
for i, cls in enumerate(COCO_CLASSES):
    print(f"  {i:2d}. {cls}")


def preprocess_image(image_input, target_size=640):
    """
    Preprocessing pipeline:
      1. Load image (from URL, file path, or numpy array)
      2. Convert to RGB
      3. Resize to target_size x target_size
      4. Normalize pixel values to [0, 1]
    Returns original image + preprocessed array
    """
    # ── Load ──────────────────────────────────────────────────────────────
    if isinstance(image_input, str) and image_input.startswith('http'):
        response = requests.get(image_input, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img_array = np.array(img)
    elif isinstance(image_input, str):
        img_array = cv2.imread(image_input)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        img_array = image_input.copy()

    original = img_array.copy()

    # ── Resize ────────────────────────────────────────────────────────────
    resized = cv2.resize(img_array, (target_size, target_size))

    # ── Normalize ─────────────────────────────────────────────────────────
    normalized = resized.astype(np.float32) / 255.0

    print(f"Original  shape : {original.shape}")
    print(f"Resized   shape : {resized.shape}")
    print(f"Pixel range     : [{normalized.min():.2f}, {normalized.max():.2f}]")

    return original, normalized


# Visualize preprocessing steps on a sample image
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
response   = requests.get(SAMPLE_URL, timeout=10)
sample_img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))

original, normalized = preprocess_image(SAMPLE_URL)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original)
axes[0].set_title(f'1. Original\n{original.shape}', fontsize=11)
axes[0].axis('off')

resized_show = cv2.resize(original, (640, 640))
axes[1].imshow(resized_show)
axes[1].set_title('2. Resized (640×640)', fontsize=11)
axes[1].axis('off')

axes[2].imshow(normalized)
axes[2].set_title('3. Normalized [0.0 – 1.0]', fontsize=11)
axes[2].axis('off')

plt.suptitle('Preprocessing Pipeline', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

def detect_and_visualize(image_input, conf_threshold=0.3, title="Detection Result"):
    """
    Run YOLOv8 detection and draw labeled bounding boxes.
    Returns detection results.
    """
    # Load image
    if isinstance(image_input, str) and image_input.startswith('http'):
        response = requests.get(image_input, timeout=10)
        img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
    else:
        img = image_input.copy()

    # Run model
    results = model.predict(img, conf=conf_threshold, verbose=False)
    result  = results[0]

    # ── Draw boxes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    colors = plt.cm.tab20.colors   # 20 distinct colors cycling

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf            = float(box.conf[0])
            cls_id          = int(box.cls[0])
            cls_name        = COCO_CLASSES[cls_id]
            color           = colors[cls_id % len(colors)]

            # Bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # Label background + text
            label = f"{cls_name}  {conf:.0%}"
            ax.text(
                x1, y1 - 6, label,
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor='none')
            )

            detections.append({'class': cls_name, 'confidence': conf,
                                'bbox': [x1, y1, x2, y2]})

    ax.set_title(f"{title}  —  {len(detections)} object(s) detected",
                 fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Print detection table
    print(f"\n{'Class':<20} {'Confidence':>12}  {'Bounding Box (x1,y1,x2,y2)'}")
    print("-" * 72)
    for d in detections:
        bb = [f"{v:.0f}" for v in d['bbox']]
        print(f"{d['class']:<20} {d['confidence']:>11.1%}  ({', '.join(bb)})")

    return detections


# Run on sample image
detections = detect_and_visualize(SAMPLE_URL, conf_threshold=0.3,
                                   title="YOLOv8 — Bus Scene")

test_images = {
    "Street Scene" : "https://ultralytics.com/images/bus.jpg",
    "Sports"       : "https://ultralytics.com/images/zidane.jpg",
}

for title, url in test_images.items():
    print(f"\n{'='*60}")
    print(f"  Image: {title}")
    print('='*60)
    detect_and_visualize(url, conf_threshold=0.3, title=title)


from google.colab import files

print("Upload any image from your computer:")
uploaded = files.upload()

for filename in uploaded.keys():
    img_array = np.array(Image.open(filename).convert('RGB'))
    print(f"\nRunning detection on: {filename}")
    detect_and_visualize(img_array, conf_threshold=0.25, title=filename)

def evaluate_model(image_inputs, conf_threshold=0.3):
    """
    Evaluate detection across multiple images.
    Reports per-class counts, avg confidence, total detections.
    """
    all_detections = []

    for label, img_input in image_inputs.items():
        if isinstance(img_input, str) and img_input.startswith('http'):
            response = requests.get(img_input, timeout=10)
            img = np.array(Image.open(BytesIO(response.content)).convert('RGB'))
        else:
            img = img_input

        results = model.predict(img, conf=conf_threshold, verbose=False)
        result  = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                all_detections.append({
                    'image'     : label,
                    'class'     : COCO_CLASSES[int(box.cls[0])],
                    'confidence': float(box.conf[0])
                })

    det_df = {}
    for d in all_detections:
        cls = d['class']
        if cls not in det_df:
            det_df[cls] = {'count': 0, 'conf_sum': 0}
        det_df[cls]['count']    += 1
        det_df[cls]['conf_sum'] += d['confidence']

    print("\n" + "=" * 50)
    print("         MODEL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"{'Class':<20} {'Count':>6}  {'Avg Confidence':>15}")
    print("-" * 50)

    sorted_classes = sorted(det_df.items(), key=lambda x: -x[1]['count'])
    for cls, v in sorted_classes:
        avg_conf = v['conf_sum'] / v['count']
        print(f"{cls:<20} {v['count']:>6}  {avg_conf:>14.1%}")

    print("-" * 50)
    print(f"{'TOTAL':<20} {len(all_detections):>6}")

    # Bar chart
    classes    = [c for c, _ in sorted_classes]
    counts     = [v['count']                    for _, v in sorted_classes]
    avg_confs  = [v['conf_sum'] / v['count']    for _, v in sorted_classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(classes, counts, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Detection Count')
    axes[0].set_title('Objects Detected per Class')
    axes[0].grid(axis='x', alpha=0.3)

    bars = axes[1].barh(classes, avg_confs, color='tomato', edgecolor='white')
    axes[1].set_xlabel('Average Confidence')
    axes[1].set_title('Average Confidence per Class')
    axes[1].set_xlim(0, 1)
    axes[1].axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
    for bar, val in zip(bars, avg_confs):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.0%}', va='center', fontsize=9)
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle('Detection Evaluation Report', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


evaluate_model(test_images)


url    = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url, timeout=10)
img    = np.array(Image.open(BytesIO(response.content)).convert('RGB'))

thresholds = [0.2, 0.5, 0.75]
fig, axes  = plt.subplots(1, 3, figsize=(18, 6))

for ax, thresh in zip(axes, thresholds):
    results = model.predict(img, conf=thresh, verbose=False)
    result  = results[0]
    vis     = img.copy()

    count = 0
    if result.boxes is not None:
        for box in result.boxes:
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf         = float(box.conf[0])
            cls_name     = COCO_CLASSES[int(box.cls[0])]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,80,80), 2)
            cv2.putText(vis, f"{cls_name} {conf:.0%}",
                        (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            count += 1

    ax.imshow(vis)
    ax.set_title(f"Threshold = {thresh}  →  {count} detections",
                 fontsize=11, fontweight='bold')
    ax.axis('off')

plt.suptitle('Effect of Confidence Threshold on Detections',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()