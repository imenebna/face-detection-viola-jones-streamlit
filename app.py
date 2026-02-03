import os
from datetime import datetime
import threading

import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


# ---------------------------
# Helpers
# ---------------------------
def hex_to_bgr(hex_color: str):
    """Convert '#RRGGBB' to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def encode_image_bytes(img_bgr, ext=".png"):
    ok, buffer = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError("Image encoding failed.")
    return buffer.tobytes()


def detect_and_draw(frame_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness=2):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors),
    )

    annotated = frame_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), rect_color_bgr, int(thickness))

    return annotated, faces


# ---------------------------
# Thread-safe storage for last webcam frame
# ---------------------------
_last_lock = threading.Lock()
_last_annotated = None
_last_count = 0


def set_last_webcam(annotated_bgr: np.ndarray, faces_count: int):
    global _last_annotated, _last_count
    with _last_lock:
        _last_annotated = annotated_bgr
        _last_count = faces_count


def get_last_webcam():
    with _last_lock:
        if _last_annotated is None:
            return None, 0
        return _last_annotated.copy(), _last_count


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Face Detection (Viola-Jones)", layout="centered")
st.title("Face Detection using Viola-Jones (Haar Cascade)")

st.markdown(
    """
### Instructions
- Choose an **input mode**:
  - **Upload image** (works everywhere)
  - **Webcam (Online/WebRTC)** (works online via your browser camera)
- Adjust:
  - **scaleFactor**
  - **minNeighbors**
  - **rectangle color**
- Save results:
  - **Save**: writes to app folder on the server
  - **Download**: saves to your device (recommended)
"""
)

# Load Haar cascade
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check your OpenCV installation.")
    st.stop()

# Settings (checkpoint requirements)
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", 1.01, 1.50, 1.10, 0.01)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 20, 5, 1)
thickness = st.sidebar.slider("Rectangle thickness", 1, 6, 2, 1)

mode = st.radio("Select input mode", ["Upload image", "Webcam (Online/WebRTC)"])

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)


# ---------------------------
# Mode 1: Upload image
# ---------------------------
if mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start detection.")
        st.stop()

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    annotated, faces = detect_and_draw(
        img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
    )

    st.write(f"Detected faces: **{len(faces)}**")
    st.image(annotated, channels="BGR", caption="Annotated result", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save annotated image"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
            cv2.imwrite(out_path, annotated)
            st.success(f"Saved as: {out_path}")

    with col2:
        st.download_button(
            "Download annotated image",
            data=encode_image_bytes(annotated, ".png"),
            file_name="annotated_faces.png",
            mime="image/png",
        )


# ---------------------------
# Mode 2: Webcam online via WebRTC
# ---------------------------
else:
    st.subheader("Webcam (Online/WebRTC)")
    st.caption("Allow camera permissions in your browser when prompted.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated, faces = detect_and_draw(
            img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
        )
        set_last_webcam(annotated, len(faces))
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="webrtc-face-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

    latest_frame, latest_count = get_last_webcam()
    st.write(f"Detected faces (latest frame): **{latest_count}**")

    st.divider()
    st.subheader("Save or download latest webcam frame")
    if latest_frame is None:
        st.info("No webcam frame captured yet. Start the webcam above.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save last webcam frame"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SAVE_DIR, f"webcam_faces_{ts}.png")
                cv2.imwrite(out_path, latest_frame)
                st.success(f"Saved as: {out_path}")
        with col2:
            st.download_button(
                "Download last webcam frame",
                data=encode_image_bytes(latest_frame, ".png"),
                file_name="webcam_annotated_faces.png",
                mime="image/png",
            )
