"""
utils.py
--------
Occupancy counting and box-drawing utility functions.
"""

import cv2
import numpy as np

COLOR_OCCUPIED = (0, 0, 255)      # red
COLOR_UNOCCUPIED = (0, 200, 0)    # green


def count_occupancy(boxes, class_names: dict) -> dict:
    occupied_count = 0
    unoccupied_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id].lower()
        if cls_name == "occupied":
            occupied_count += 1
        elif cls_name == "unoccupied":
            unoccupied_count += 1

    total_seats = occupied_count + unoccupied_count

    if total_seats == 0:
        return {
            "occupied": 0,
            "unoccupied": 0,
            "total_seats": 0,
            "occupancy_percent": None,
            "warning": "No seats detected in this frame."
        }

    occupancy_percent = round((occupied_count / total_seats) * 100, 1)

    return {
        "occupied": occupied_count,
        "unoccupied": unoccupied_count,
        "total_seats": total_seats,
        "occupancy_percent": occupancy_percent,
        "warning": None
    }


def draw_detections(image: np.ndarray, boxes, class_names: dict) -> np.ndarray:
    annotated = image.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLOR_OCCUPIED if cls_name.lower() == "occupied" else COLOR_UNOCCUPIED
        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated