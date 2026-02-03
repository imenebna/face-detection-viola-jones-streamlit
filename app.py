import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

# Live video (optional). If not installed or not supported, app falls back to st.camera_input.
try:
    import av  # noqa: F401
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def hex_to_bgr(hex_color: str):
    """Convert '#RRGGBB' to OpenCV BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def detect_and_draw(frame_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness=2):
    """Detect faces and draw rectangles; returns (annotated_bgr, faces)."""
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


def encode_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG.")
    return buf.tobytes()


def set_last_result(img_bgr: np.ndarray, faces_count: int):
    st.session_state["last_annotated_bgr"] = img_bgr
    st.session_state["last_faces_count"] = int(faces_count)


def get_last_result():
    return st.session_state.get("last_annotated_bgr"), st.session_state.get("last_faces_count", 0)


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Face Detection (Viola-Jones)", layout="wide")
st.title("Face Detection using Viola–Jones (Haar Cascade)")

# ✅ Checkpoint requirement: instructions in UI
st.markdown(
    """
### Instructions
1) Choose **Upload image** or **Webcam**.  
2) Adjust detection sliders:
- **scaleFactor**: smaller values can detect more faces but may be slower.
- **minNeighbors**: higher values reduce false positives but may miss faces.
3) Choose the **rectangle color**.
4) Use **Download** to save the annotated result to your device.
"""
)

# Load Haar cascade (portable)
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check your OpenCV installation.")
    st.stop()

# Sidebar controls (✅ color picker + sliders)
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")  # ✅ requirement
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", 1.01, 1.50, 1.10, 0.01)  # ✅ requirement
min_neighbors = st.sidebar.slider("minNeighbors", 1, 20, 5, 1)            # ✅ requirement
thickness = st.sidebar.slider("Rectangle thickness", 1, 6, 2, 1)

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)

if "last_annotated_bgr" not in st.session_state:
    st.session_state["last_annotated_bgr"] = None
if "last_faces_count" not in st.session_state:
    st.session_state["last_faces_count"] = 0

mode = st.radio("Select input mode", ["Upload image", "Webcam"], horizontal=True)

# ============================================================
# MODE 1: Upload image (fixed: no PIL, no st.image kwargs issues)
# ============================================================
if mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run face detection.")
    else:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the uploaded image.")
            st.stop()

        annotated_bgr, faces = detect_and_draw(
            img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
        )
        set_last_result(annotated_bgr, len(faces))

        st.write(f"Detected faces: **{len(faces)}**")

        # Robust display: RGB numpy array (no PIL, no problematic kwargs)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Annotated result")

# ============================================================
# MODE 2: Webcam
# - Preferred: LIVE video via WebRTC (if available)
# - Fallback: st.camera_input capture (always works online)
# ============================================================
else:
    st.subheader("Webcam")

    # -------- Live video mode (WebRTC) --------
    if WEBRTC_AVAILABLE:
        st.caption("Live video detection (WebRTC). Allow camera permissions when prompted.")

        # Optional TURN from Streamlit Secrets (flat keys)
        # Put these in Streamlit Cloud Secrets if you want TURN:
        # TURN_USERNAME, TURN_PASSWORD, TURN_URL_1, TURN_URL_2
        turn_user = st.secrets.get("TURN_USERNAME", "")
        turn_pass = st.secrets.get("TURN_PASSWORD", "")
        turn_url_1 = st.secrets.get("TURN_URL_1", "")
        turn_url_2 = st.secrets.get("TURN_URL_2", "")
        turn_urls = [u for u in [turn_url_1, turn_url_2] if u]

        ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
        if turn_user and turn_pass and turn_urls:
            ice_servers.insert(0, {"urls": turn_urls, "username": turn_user, "credential": turn_pass})

        class FaceProcessor:
            def __init__(self):
                self.last_faces = 0

            def recv(self, frame):
                img_bgr = frame.to_ndarray(format="bgr24")

                annotated_bgr, faces = detect_and_draw(
                    img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
                )

                self.last_faces = len(faces)
                set_last_result(annotated_bgr, self.last_faces)

                return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

        ctx = webrtc_streamer(
            key="face-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={"video": True, "audio": False},
            video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True},
            async_processing=False,
            video_processor_factory=FaceProcessor,
        )

        if ctx.video_processor:
            st.write(f"Detected faces (latest frame): **{ctx.video_processor.last_faces}**")

        st.divider()
        st.caption("If live video stays black on your network, use the capture fallback below.")

    else:
        st.warning("Live video mode requires streamlit-webrtc. Using camera capture fallback.")

    # -------- Capture fallback (always online reliable) --------
    cam_file = st.camera_input("Capture a frame (fallback that works reliably online)")

    if cam_file is not None:
        file_bytes = np.frombuffer(cam_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the captured image. Please try again.")
        else:
            annotated_bgr, faces = detect_and_draw(
                img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
            )
            set_last_result(annotated_bgr, len(faces))

            st.write(f"Detected faces (captured frame): **{len(faces)}**")
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Annotated captured frame")

# ============================================================
# SAVE / DOWNLOAD (Checkpoint: save to user device + cv2.imwrite used)
# ============================================================
st.divider()
st.subheader("Save / Download last result")

latest_bgr, latest_count = get_last_result()

if latest_bgr is None:
    st.info("No annotated result yet. Upload an image or use the webcam first.")
else:
    st.write(f"Detected faces (last result): **{latest_count}**")

    col1, col2 = st.columns(2)

    with col1:
        # Uses cv2.imwrite (hint) - optional server save
        if st.button("Save on server (optional)"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
            cv2.imwrite(out_path, latest_bgr)
            st.success(f"Saved on server as: {out_path}")

    with col2:
        # ✅ Required: save to user's device
        st.download_button(
            label="Download annotated image (to your device)",
            data=encode_png_bytes(latest_bgr),
            file_name="annotated_faces.png",
            mime="image/png",
        )
