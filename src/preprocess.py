"""
preprocess.py
--------------
Handles image preprocessing for the cinema occupancy project.
Brightens dark frames using CLAHE before they go into the model.
"""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def is_low_light(image: np.ndarray, threshold: float = 90.0) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < threshold


def preprocess_frame(image: np.ndarray, force_clahe: bool = False) -> tuple:
    if force_clahe or is_low_light(image):
        return apply_clahe(image), True
    return image, False