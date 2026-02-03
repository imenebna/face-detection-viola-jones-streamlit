import os
import cv2
import numpy as np
import streamlit as st
from datetime import datetime

# ===============================
# Utilities
# ===============================

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(img_bgr, scaleFactor, minNeighbors):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_bgr, faces

# ===============================
# Streamlit setup
# ===============================

st.set_page_config(page_title="Face Detection – Viola Jones")
st.title("Face Detection (Viola–Jones)")

# ===============================
# Load cascade
# ===============================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error("Failed to load Haar Cascade")
    st.stop()

# ===============================
# Sidebar controls
# ===============================

scaleFactor = st.sidebar.slider("scaleFactor", 1.01, 1.5, 1.1, 0.01)
minNeighbors = st.sidebar.slider("minNeighbors", 1, 10, 5)

mode = st.radio(
    "Select input mode",
    ["Upload image", "Webcam (local only)"]
)

# ===============================
# IMAGE UPLOAD (100% WORKING)
# ===============================

if mode == "Upload image":
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if file is not None:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Invalid image file.")
            st.stop()

        annotated, faces = detect_faces(
            img_bgr.copy(),
            scaleFactor,
            minNeighbors
        )

        annotated_rgb = bgr_to_rgb(annotated)

        st.success(f"Detected faces: {len(faces)}")
        st.image(annotated_rgb, caption="Annotated result")

        # Download
        _, buffer = cv2.imencode(".png", annotated)
        st.download_button(
            "Download image",
            buffer.tobytes(),
            file_name="faces.png",
            mime="image/png"
        )

# ===============================
# WEBCAM (LOCAL ONLY – HONEST)
# ===============================

else:
    st.warning(
        "⚠ Webcam works ONLY when running Streamlit locally.\n"
        "Streamlit Cloud free tier cannot reliably access webcams."
    )

    start = st.button("Start webcam")
    stop = st.button("Stop webcam")

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True
    if stop:
        st.session_state.run_cam = False

    frame_box = st.empty()

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Cannot open webcam.")
            st.session_state.run_cam = False
        else:
            while st.session_state.run_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, faces = detect_faces(
                    frame,
                    scaleFactor,
                    minNeighbors
                )

                frame_rgb = bgr_to_rgb(annotated)
                frame_box.image(frame_rgb)

            cap.release()
