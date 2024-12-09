import torch
import os

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 5

FASTER_RCNN_MODEL_PATH = os.path.join("models", "models/fasterrcnn_coco_apples_gpu.pth")
