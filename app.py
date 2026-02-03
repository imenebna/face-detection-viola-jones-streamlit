import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image


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


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray to PIL RGB."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


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
st.title("Face Detection using Viola-Jones (Haar Cascade)")

st.markdown(
    """
### Instructions
1. Choose **Upload image** or **Webcam (Online)**.
2. Adjust detection sliders:
   - **scaleFactor**: smaller values can detect more faces but may be slower.
   - **minNeighbors**: higher values reduce false positives but may miss faces.
3. Choose the **rectangle color**.
4. Use **Download** to save the annotated result to your device.
"""
)

# Load Haar cascade (portable)
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check your OpenCV installation.")
    st.stop()

# Sidebar controls (checkpoint requirements)
st.sidebar.header("Detection Settings")
rect_hex = st.sidebar.color_picker("Rectangle color", value="#00FF00")
rect_color_bgr = hex_to_bgr(rect_hex)

scale_factor = st.sidebar.slider("scaleFactor", min_value=1.01, max_value=1.50, value=1.10, step=0.01)
min_neighbors = st.sidebar.slider("minNeighbors", min_value=1, max_value=20, value=5, step=1)
thickness = st.sidebar.slider("Rectangle thickness", min_value=1, max_value=6, value=2, step=1)

SAVE_DIR = "saved_faces"
ensure_dir(SAVE_DIR)

if "last_annotated_bgr" not in st.session_state:
    st.session_state["last_annotated_bgr"] = None
if "last_faces_count" not in st.session_state:
    st.session_state["last_faces_count"] = 0

input_mode = st.radio("Select input mode", ["Upload image", "Webcam (Online)"], horizontal=False)

# ============================================================
# MODE 1: Upload image
# ============================================================
if input_mode == "Upload image":
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
        st.image(bgr_to_pil(annotated_bgr), caption="Annotated result", use_container_width=True)

# ============================================================
# MODE 2: Webcam ONLINE (reliable) using st.camera_input
# ============================================================
else:
    st.subheader("Webcam (Online)")
    st.caption(
        "This uses Streamlit’s built-in camera capture (works reliably online). "
        "Capture a frame, then face detection runs on that image."
    )

    cam_file = st.camera_input("Take a photo with your webcam")

    if cam_file is None:
        st.info("Click the camera button above, allow permissions, then capture a photo.")
    else:
        # Camera image bytes -> OpenCV image
        file_bytes = np.frombuffer(cam_file.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not read the camera image. Please capture again.")
            st.stop()

        annotated_bgr, faces = detect_and_draw(
            img_bgr, face_cascade, scale_factor, min_neighbors, rect_color_bgr, thickness
        )

        set_last_result(annotated_bgr, len(faces))

        st.write(f"Detected faces: **{len(faces)}**")
        st.image(bgr_to_pil(annotated_bgr), caption="Annotated result", use_container_width=True)

# ============================================================
# SAVE / DOWNLOAD (checkpoint requirement: save to user's device)
# ============================================================
st.divider()
st.subheader("Save / Download last result")

latest_bgr, latest_count = get_last_result()

if latest_bgr is None:
    st.info("No annotated result yet. Upload an image or capture a webcam photo first.")
else:
    st.write(f"Detected faces (last result): **{latest_count}**")

    col1, col2 = st.columns(2)

    with col1:
        # Save on server (optional, still using cv2.imwrite per hint)
        if st.button("Save on server (optional)"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(SAVE_DIR, f"faces_{ts}.png")
            cv2.imwrite(out_path, latest_bgr)
            st.success(f"Saved on server as: {out_path}")

    with col2:
        # Save to user's device (required)
        st.download_button(
            label="Download annotated image (to your device)",
            data=encode_png_bytes(latest_bgr),
            file_name="annotated_faces.png",
            mime="image/png",
        )
