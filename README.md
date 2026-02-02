# Face Detection App (Viola-Jones) Checkpoint

This project is a Streamlit app that performs face detection using OpenCV Haar Cascades (Viola–Jones),
with checkpoint features:
- UI instructions
- Save annotated images
- Choose rectangle color
- Adjust `minNeighbors`
- Adjust `scaleFactor`

## Files
- `app.py` : Streamlit application
- `requirements.txt` : dependencies

## Run (Windows / VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Run (macOS / Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- For saving to your device, use the **Download** button in the app.
- Webcam mode works best when running locally.
