import os
from datetime import datetime
import threading

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


# ----------------------------
# Checkpoint helpers
# ----------------------------
def hex_to_bgr(hex_color: str):
    """#RRGGBB -> (B, G, R)"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return buf.tobytes()


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Most robust for st.image on Streamlit Cloud."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    return Image.fromarray(img_rgb)


def detect_and_draw(img_bgr: np.ndarray,
                    face_cascade,
                    scale_factor: float,
                    min_neighbors: int,
                    rect_color_bgr: tuple,
                    thickness: int):
    """Viola-Jones face detection + draw rectangles."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=float(scale_factor),
        minNeighbors=int(min_neighbors)
    )

    annotated = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), rect_color_bgr, int(thickness))
    return annotated, faces


# ----------------------------
# Thread-safe “last webcam frame” store
# ----------------------------
_last_lock = threading.Lock()
_last_webcam_bgr = None
_last_webcam_count = 0


def set_last_webcam(img_bgr: np.ndarray, count: int):
    global _last_webcam_bgr, _last_webcam_count
    with _last_lock:
        _last_webcam_bgr = img_bgr
        _last_webcam_count = count


def get_last_webcam():
    with _last_lock:
        if _last_webcam_bgr is None:
            return None, 0
        return _last_webcam_bgr.copy(), _last_webcam_count


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Face Detection (Viola-Jones)", layout="centered")
st.title("Face Detection using Viola-Jones (Haar Cascade)")

# ✅ Requirement: instructions in UI
st.markdown(
    """
### How to use this app
1) Choose **Upload image** or **Webcam (Online/WebRTC)**.  
2) Adjust detection sliders:
- **scaleFactor**: smaller values can detect more faces but may be slower.
- **minNeighbors**: higher values reduce false positives but may miss faces.
3) Choose the **rectangle color**.
4) Click **Download** to save the annotated result **to your device**.
"""
)

# Load Haar cascade
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check OpenCV installation.")
    st.stop()

# Sidebar controls (✅ requirement sliders + color picker)
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")  # ✅ requirement
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", 1.01, 1.50, 1.10, 0.01)  # ✅ requirement
min_neighbors = st.sidebar.slider("minNeighbors", 1, 20, 5, 1)           # ✅ requirement
thickness = st.sidebar.slider("Rectangle thickness", 1, 6, 2, 1)

mode = st.radio("Select input mode", ["Upload image", "Webcam (Online/WebRTC)"])

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)


# ============================================================
# MODE 1: Upload image (fully functional on Cloud)
# ============================================================
if mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start detection.")
        st.stop()

    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    annotated_bgr, faces = detect_and_draw(
        img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
    )

    st.write(f"Detected faces: **{len(faces)}**")

    # Display robustly (PIL avoids many Streamlit Cloud ndarray issues)
    st.image(bgr_to_pil(annotated_bgr), caption="Annotated result")

    # ✅ requirement: save to user's device
    png_bytes = encode_png_bytes(annotated_bgr)
    st.download_button(
        label="Download annotated image (to your device)",
        data=png_bytes,
        file_name="annotated_faces.png",
        mime="image/png"
    )

    # (Optional) server save too (not required by checkpoint but useful)
    if st.button("Save on server (optional)"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
        cv2.imwrite(out_path, annotated_bgr)
        st.success(f"Saved on server as: {out_path}")


# ============================================================
# MODE 2: Webcam online via WebRTC
# ============================================================
else:
    st.subheader("Webcam (Online/WebRTC)")
    st.caption("Allow camera access in your browser. For best reliability on Streamlit Cloud, configure TURN (see below).")

    # WebRTC routing: STUN (often ok) + TURN (makes it reliable on many networks)
    turn_user = st.secrets.get("TURN_USERNAME", "")
    turn_pass = st.secrets.get("TURN_PASSWORD", "")
    turn_url_1 = st.secrets.get("TURN_URL_1", "")
    turn_url_2 = st.secrets.get("TURN_URL_2", "")
    
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    turn_urls = [u for u in [turn_url_1, turn_url_2] if u]
    if turn_user and turn_pass and turn_urls:
        ice_servers.insert(
            0,
            {"urls": turn_urls, "username": turn_user, "credential": turn_pass}
        )


    class FaceProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")

            annotated_bgr, faces = detect_and_draw(
                img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
            )

            set_last_webcam(annotated_bgr, len(faces))
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    webrtc_streamer(
        key="face-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=FaceProcessor,
        async_processing=False,
        video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True},
    )

    latest_bgr, latest_count = get_last_webcam()
    st.write(f"Detected faces (latest frame): **{latest_count}**")

    st.divider()
    st.subheader("Save / Download last webcam frame")

    if latest_bgr is None:
        st.info("No webcam frame captured yet. Click Start in the webcam player above.")
    else:
        # ✅ requirement: save to user's device from webcam too
        st.download_button(
            label="Download last webcam frame (to your device)",
            data=encode_png_bytes(latest_bgr),
            file_name="webcam_annotated_faces.png",
            mime="image/png"
        )

        # Optional server save
        if st.button("Save last webcam frame on server (optional)"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"webcam_faces_{ts}.png")
            cv2.imwrite(out_path, latest_bgr)
            st.success(f"Saved on server as: {out_path}")
