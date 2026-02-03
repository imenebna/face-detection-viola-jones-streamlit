import os
from datetime import datetime
import threading

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


# =========================================================
# Checkpoint features included:
# 1) Instructions in UI (st.markdown)
# 2) Save annotated images (cv2.imwrite) + download to device
# 3) Rectangle color picker (st.color_picker)
# 4) minNeighbors slider (st.slider)
# 5) scaleFactor slider (st.slider)
#
# Webcam works on Streamlit Cloud using WebRTC (streamlit-webrtc)
# =========================================================


# ---------------------------
# Helpers
# ---------------------------
def hex_to_bgr(hex_color: str):
    """Convert Streamlit hex color '#RRGGBB' to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def encode_image_bytes(img_bgr, ext=".png"):
    """Encode BGR image as bytes for downloading."""
    ok, buffer = cv2.imencode(ext, img_bgr)
    if not ok:
        raise ValueError("Image encoding failed.")
    return buffer.tobytes()


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


# ---------------------------
# Global shared state for webcam frames (thread-safe)
# ---------------------------
_last_lock = threading.Lock()
_last_annotated_bgr = None
_last_faces_count = 0


def set_last_frame(img_bgr: np.ndarray, faces_count: int):
    global _last_annotated_bgr, _last_faces_count
    with _last_lock:
        _last_annotated_bgr = img_bgr
        _last_faces_count = faces_count


def get_last_frame():
    with _last_lock:
        if _last_annotated_bgr is None:
            return None, 0
        return _last_annotated_bgr.copy(), _last_faces_count


# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Face Detection (Viola-Jones)", layout="centered")
st.title("Face Detection using Viola-Jones (Haar Cascade)")

st.markdown(
    """
### Instructions
- Choose an **input mode**:
  - **Upload image**: works everywhere (including Streamlit Cloud).
  - **Webcam (Cloud/WebRTC)**: works online using browser camera access.
- Tune detection with:
  - **scaleFactor**: smaller steps (e.g., 1.05) can detect more faces but may be slower.
  - **minNeighbors**: higher values reduce false positives but may miss faces.
- Pick the **rectangle color** and thickness.
- Use **Save** (writes to the machine running the app) or **Download** (saves to your device).
"""
)

# Load Haar cascade safely
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check your OpenCV installation.")
    st.stop()

# Sidebar checkpoint controls
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", min_value=1.01, max_value=1.50, value=1.10, step=0.01)
min_neighbors = st.sidebar.slider("minNeighbors", min_value=1, max_value=20, value=5, step=1)
thickness = st.sidebar.slider("Rectangle thickness", min_value=1, max_value=6, value=2, step=1)

input_mode = st.radio("Select input mode", ["Upload image", "Webcam (Cloud/WebRTC)"])

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)


# =========================================================
# Mode 1: Upload image
# =========================================================
if input_mode == "Upload image":
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
        img_bgr,
        face_cascade,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        rect_color_bgr=rect_color_bgr,
        thickness=thickness
    )

    st.write(f"Detected faces: **{len(faces)}**")

    if annotated is None or not isinstance(annotated, np.ndarray):
        st.error("Annotated image is invalid.")
        st.stop()

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
            st.success(f"Saved as: {out_path}")

    with col2:
        img_bytes = encode_image_bytes(annotated, ext=".png")
        st.download_button(
            label="Download annotated image",
            data=img_bytes,
            file_name="annotated_faces.png",
            mime="image/png"
        )


# =========================================================
# Mode 2: Webcam on Streamlit Cloud using WebRTC
# =========================================================
else:
    st.subheader("Webcam (Cloud/WebRTC)")
    st.caption("If prompted by your browser, allow camera access.")

    # Frame processing callback (runs in a background thread)
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        annotated, faces = detect_and_draw(
            img_bgr,
            face_cascade,
            scale_factor=float(scale_factor),
            min_neighbors=int(min_neighbors),
            rect_color_bgr=rect_color_bgr,
            thickness=int(thickness)
        )

        set_last_frame(annotated, len(faces))

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="face-detection-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
        async_processing=True,
    )

    # Show latest count (polled from shared state)
    latest_frame, latest_count = get_last_frame()
    st.write(f"Detected faces (latest frame): **{latest_count}**")

    st.divider()
    st.subheader("Save or download latest annotated webcam frame")

    if latest_frame is None:
        st.info("No frame captured yet. Start the webcam above.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save last webcam frame"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SAVE_DIR, f"webcam_faces_{ts}.png")
                cv2.imwrite(out_path, latest_frame)
                st.success(f"Saved as: {out_path}")

        with col2:
            img_bytes = encode_image_bytes(latest_frame, ext=".png")
            st.download_button(
                label="Download last webcam frame",
                data=img_bytes,
                file_name="webcam_annotated_faces.png",
                mime="image/png"
            )


     
