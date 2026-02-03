import os
from datetime import datetime
import threading

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av


# ---------------------------
# Helpers
# ---------------------------
def hex_to_bgr(hex_color: str):
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
    scale_factor = float(scale_factor)
    min_neighbors = int(min_neighbors)
    thickness = int(thickness)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    annotated = frame_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), rect_color_bgr, thickness)

    return annotated, faces


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Most compatible display path for Streamlit Cloud."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Ensure contiguous uint8 (avoids rare Streamlit type errors)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    return Image.fromarray(img_rgb)


# ---------------------------
# Thread-safe last webcam frame storage
# ---------------------------
_last_lock = threading.Lock()
_last_bgr = None
_last_count = 0


def set_last_webcam(img_bgr: np.ndarray, count: int):
    global _last_bgr, _last_count
    with _last_lock:
        _last_bgr = img_bgr
        _last_count = count


def get_last_webcam():
    with _last_lock:
        if _last_bgr is None:
            return None, 0
        return _last_bgr.copy(), _last_count


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
  - **Webcam (Online/WebRTC)** (works online in the browser)
- Adjust detection:
  - **scaleFactor**
  - **minNeighbors**
- Choose the **rectangle color**.
- Save results:
  - **Save**: writes to server folder (`saved_faces/`)
  - **Download**: saves to your device
"""
)

# Load cascade
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check OpenCV installation.")
    st.stop()

# Sidebar controls (checkpoint requirements)
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", 1.01, 1.50, 1.10, 0.01)
min_neighbors = st.sidebar.slider("minNeighbors", 1, 20, 5, 1)
thickness = st.sidebar.slider("Rectangle thickness", 1, 6, 2, 1)

mode = st.radio("Select input mode", ["Upload image", "Webcam (Online/WebRTC)"])

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)


# =========================================================
# MODE 1: Upload image (fixes st.image TypeError by using PIL)
# =========================================================
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

    annotated_bgr, faces = detect_and_draw(
        img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
    )

    st.write(f"Detected faces: **{len(faces)}**")

    # Display as PIL image (most stable on Streamlit Cloud)
    st.image(bgr_to_pil(annotated_bgr), caption="Annotated result", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save annotated image"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
            cv2.imwrite(out_path, annotated_bgr)
            st.success(f"Saved as: {out_path}")

    with col2:
        st.download_button(
            "Download annotated image",
            data=encode_image_bytes(annotated_bgr, ".png"),
            file_name="annotated_faces.png",
            mime="image/png",
        )


# =========================================================
# MODE 2: Webcam WebRTC (more stable processor pattern)
# =========================================================
else:
    st.subheader("Webcam (Online/WebRTC)")
    st.caption("Allow camera permissions. If it stays black, try Chrome + another network (WebRTC can be blocked).")

    RTC_CONFIGURATION = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
        ]
    }

    class FaceProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            annotated_bgr, faces = detect_and_draw(
                img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
            )

            set_last_webcam(annotated_bgr, len(faces))
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    webrtc_streamer(
        key="webrtc-face-detect",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=FaceProcessor,
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "muted": True,
            "playsInline": True,   # important on some browsers
            "controls": False
        },
    )

    latest_bgr, latest_count = get_last_webcam()
    st.write(f"Detected faces (latest frame): **{latest_count}**")

    st.divider()
    st.subheader("Save or download latest webcam frame")

    if latest_bgr is None:
        st.info("No webcam frame captured yet. Click Start in the webcam component above.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save last webcam frame"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(SAVE_DIR, f"webcam_faces_{ts}.png")
                cv2.imwrite(out_path, latest_bgr)
                st.success(f"Saved as: {out_path}")

        with col2:
            st.download_button(
                "Download last webcam frame",
                data=encode_image_bytes(latest_bgr, ".png"),
                file_name="webcam_annotated_faces.png",
                mime="image/png",
            )
