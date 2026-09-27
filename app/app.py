"""
app.py
------
The Streamlit dashboard.
Run with: streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from src.model import load_model, run_inference
from src.preprocess import preprocess_frame
from src.utils import count_occupancy, draw_detections

st.set_page_config(page_title="Cinema Occupancy Detection", layout="wide")
st.title("🎬 Cinema Occupancy Detection")
st.caption("YOLO26-based seat occupancy detection, with low-light preprocessing.")

@st.cache_resource
def get_model():
    return load_model()

model, is_fine_tuned = get_model()

if not is_fine_tuned:
    st.warning(
        "⚠️ Using the pretrained fallback model (yolo26n.pt), not your fine-tuned weights. "
        "This won't correctly detect 'occupied'/'unoccupied' seats until you train the model "
        "and place best.pt in models/trained/."
    )

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
force_low_light_mode = st.sidebar.checkbox("Force low-light enhancement (CLAHE)")

uploaded_file = st.file_uploader("Upload a cinema hall image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    processed_image, was_enhanced = preprocess_frame(image_bgr, force_clahe=force_low_light_mode)

    result = run_inference(model, processed_image, confidence=confidence)
    class_names = result.names

    annotated_image = draw_detections(processed_image, result.boxes, class_names)
    stats = count_occupancy(result.boxes, class_names)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        title = "Detection Result"
        if was_enhanced:
            title += " (low-light enhanced)"
        st.subheader(title)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()
    st.subheader("Occupancy Summary")

    if stats["warning"]:
        st.info(stats["warning"])
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Occupied Seats", stats["occupied"])
        m2.metric("Unoccupied Seats", stats["unoccupied"])
        m3.metric("Occupancy %", f"{stats['occupancy_percent']}%")
else:
    st.info("Upload an image to run detection.")