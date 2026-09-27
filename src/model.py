"""
model.py
--------
Loads the trained YOLO model and runs detection on an image.
"""

import os
from ultralytics import YOLO

DEFAULT_WEIGHTS_PATH = "models/trained/best.pt"
FALLBACK_WEIGHTS_PATH = "yolo26n.pt"


def load_model(weights_path: str = DEFAULT_WEIGHTS_PATH) -> tuple:
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        return model, True
    else:
        model = YOLO(FALLBACK_WEIGHTS_PATH)
        return model, False


def run_inference(model, image, confidence: float = 0.4):
    results = model(image, conf=confidence, verbose=False)
    return results[0]