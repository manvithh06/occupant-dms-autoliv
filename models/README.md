# Models Directory

Place trained model weights here.  Large binary files are excluded from Git
(see `.gitignore`); use the instructions below to obtain them.

---

## Default Model

The system ships with `yolov8n.pt` (YOLOv8 Nano, COCO-pretrained, ~6 MB)
bundled in `src/`.  It is used as a fallback when no fine-tuned model is found
here.

**Limitation**: The COCO model cannot distinguish adults, children, and infants.
A bounding-box **size heuristic** is applied as an interim measure
(see `src/occupant/occupant_detector.py → _classify_person()`).

---

## Fine-Tuned Occupant Model (Recommended)

For production-grade adult / child / infant classification, train or fine-tune
a YOLOv8 model on a vehicle-occupant dataset and place the weights here as:

```
models/occupant_dms.pt
```

The system automatically prefers this file over the bundled `yolov8n.pt`
(see `src/config.py → DMSConfig.model_path`).

### Recommended Datasets

| Dataset | Classes | Notes |
|---------|---------|-------|
| [CDS Child Detection Dataset](https://universe.roboflow.com) | adult, child, infant | Vehicle interior images |
| [COCO People](https://cocodataset.org) | person | Base fine-tuning data |
| Custom labelled frames | adult, child, infant, pet | Use Roboflow / Label Studio |

### Training (YOLOv8)

```bash
# Install Ultralytics
pip install ultralytics

# Fine-tune on your dataset (adjust --data path)
yolo detect train \
    model=yolov8n.pt \
    data=dataset/data.yaml \
    epochs=100 \
    imgsz=640 \
    project=models \
    name=occupant_dms

# The best weights will be saved to:
#   models/occupant_dms/weights/best.pt
# Copy to models/occupant_dms.pt
```

---

## ONNX Export (Embedded / Edge Deployment)

For faster inference on edge hardware (Jetson, Raspberry Pi):

```bash
yolo export model=models/occupant_dms.pt format=onnx opset=12
# Produces models/occupant_dms.onnx
```

Update `config.yaml`:
```yaml
model_path: models/occupant_dms.onnx
```
