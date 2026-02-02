import os
import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st


def hex_to_bgr(hex_color: str):
    """Convert Streamlit hex color '#RRGGBB' to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def detect_and_draw(frame_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness=2):
    """Detect faces and draw rectangles; returns (annotated_frame, faces)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    annotated = frame_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), rect_color_bgr, thickness)

    return annotated, faces


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def encode_image_bytes(img_bgr, ext=".png"):
    """Encode BGR image as bytes for downloading."""
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        raise ValueError("Image encoding failed.")
    return buffer.tobytes()


st.set_page_config(page_title="Face Detection (Viola-Jones)", layout="centered")
st.title("Face Detection using Viola-Jones (Haar Cascade)")

st.markdown(
    """
### Instructions
- Choose an **input mode**: upload an image or use your **webcam** (works best when running locally).
- Tune detection with:
  - **scaleFactor**: smaller steps (e.g., 1.05) can detect more faces but may be slower.
  - **minNeighbors**: higher values reduce false positives but may miss faces.
- Pick the **rectangle color**.
- Use **Save & Download** to save the annotated result to your device.
"""
)

cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check your OpenCV installation.")
    st.stop()

st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", min_value=1.01, max_value=1.50, value=1.10, step=0.01)
min_neighbors = st.sidebar.slider("minNeighbors", min_value=1, max_value=20, value=5, step=1)
thickness = st.sidebar.slider("Rectangle thickness", min_value=1, max_value=6, value=2, step=1)

input_mode = st.radio("Select input mode", ["Upload image", "Webcam (local)"])

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)

if "last_annotated" not in st.session_state:
    st.session_state.last_annotated = None
if "last_faces_count" not in st.session_state:
    st.session_state.last_faces_count = 0


if input_mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        annotated, faces = detect_and_draw(
            img_bgr,
            face_cascade,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            rect_color_bgr=rect_color_bgr,
            thickness=thickness
        )

        st.session_state.last_annotated = annotated
        st.session_state.last_faces_count = len(faces)

        st.write(f"Detected faces: **{len(faces)}**")
        if annotated is None or not isinstance(annotated, np.ndarray):
        st.error("Annotated image is invalid.")
        else:
        st.image(
            annotated,
            channels="BGR",
            caption="Annotated result",
            use_container_width=True
        )



        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save annotated image"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
                cv2.imwrite(out_path, annotated)
                st.success(f"Saved on server as: {out_path}")

        with col2:
            if st.session_state.last_annotated is not None:
                img_bytes = encode_image_bytes(st.session_state.last_annotated, ext=".png")
                st.download_button(
                    label="Download annotated image",
                    data=img_bytes,
                    file_name="annotated_faces.png",
                    mime="image/png"
                )

else:
    st.warning(
        "Webcam mode works best when you run Streamlit locally on your computer. "
        "If your browser/server cannot access the webcam, use 'Upload image' instead."
    )

    start = st.button("Start webcam")
    stop = st.button("Stop webcam")

    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True
    if stop:
        st.session_state.run_cam = False

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Try another camera index or use Upload image mode.")
            st.session_state.run_cam = False
        else:
            while st.session_state.run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                annotated, faces = detect_and_draw(
                    frame,
                    face_cascade,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    rect_color_bgr=rect_color_bgr,
                    thickness=thickness
                )

                st.session_state.last_annotated = annotated
                st.session_state.last_faces_count = len(faces)

                info_placeholder.write(f"Detected faces: **{len(faces)}**")
                frame_placeholder.image(annotated, channels="BGR", caption="Webcam stream (annotated)", use_container_width=True)
                time.sleep(0.03)

            cap.release()

    if st.session_state.last_annotated is not None:
        st.divider()
        st.subheader("Save or download last annotated frame")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save last frame"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SAVE_DIR, f"webcam_faces_{ts}.png")
                cv2.imwrite(out_path, st.session_state.last_annotated)
                st.success(f"Saved on server as: {out_path}")

        with col2:
            img_bytes = encode_image_bytes(st.session_state.last_annotated, ext=".png")
            st.download_button(
                label="Download last frame",
                data=img_bytes,
                file_name="webcam_annotated_faces.png",
                mime="image/png"
            )
    else:
        st.info("No annotated frame yet. Start the webcam or upload an image first.")
    if annotated is None or not isinstance(annotated, np.ndarray):
    st.error("Annotated image is invalid.")
else:
    st.image(
        annotated,
        channels="BGR",
        caption="Annotated result",
        use_container_width=True
    )

