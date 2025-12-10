import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io
import os

# --- Configuration ---
MODEL_OPTIONS = {
    "ANPR (Automatic Number Plate Recognition)": "yolo_ANPR.pt",
    "ATCC (Automatic Traffic Count and Classification)": "yolo_ATCC.pt",
}

ATCC_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

ANPR_CLASSES = ["license plate"]

# --- Streamlit Setup ---
st.set_page_config(
    page_title="YOLO Object Detection App",
    layout="wide"
)

st.title("🚗 YOLO Model Selector & Detector")
st.markdown("Upload an image or video and choose a model to run object detection.")

# --- Model Loading (Caching for efficiency) ---
@st.cache_resource
def load_model(model_path):
    """Loads a YOLO model and caches it."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# --- Main Interface ---

col1, col2 = st.columns([1, 2])

with col1:
    # Model Selection Dropdown
    model_choice = st.selectbox(
        "Select YOLO Model",
        list(MODEL_OPTIONS.keys())
    )
    
    model_path = MODEL_OPTIONS[model_choice]
    model = load_model(model_path)

    if model is None:
        st.stop()

    st.sidebar.header("Configuration")
    confidence = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload an Image or Video",
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi']
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png']:
            input_type = 'image'
        elif file_extension in ['mp4', 'mov', 'avi']:
            input_type = 'video'
        else:
            st.error("Unsupported file type.")
            st.stop()

        process_button = st.button("Run Detection")

with col2:
    if uploaded_file and 'process_button' in locals() and process_button:
        with st.spinner(f"Running {model_choice} detection on your {input_type}..."):
            
            if input_type == 'image':
                try:
                    image = Image.open(uploaded_file)
                    
                    results = model.predict(source=image, conf=confidence, save=False)
                    
                    res_plotted = results[0].plot()[:, :, ::-1]
                    result_image = Image.fromarray(res_plotted)
                    
                    st.header("🖼️ Detection Result")
                    st.image(result_image, caption="Image with Bounding Boxes", use_container_width=True)
                    
                    detected_objects = results[0].boxes.cls.tolist()
                    class_names = model.names
                    class_counts = {}
                    
                    for cls in detected_objects:
                        name = class_names.get(int(cls), f"Unknown Class {int(cls)}")
                        class_counts[name] = class_counts.get(name, 0) + 1

                    st.subheader("Summary of Findings")
                    if class_counts:
                        st.markdown(f"**Total objects detected:** {len(detected_objects)}")
                        st.dataframe(
                            {"Class": list(class_counts.keys()), "Count": list(class_counts.values())},
                            use_container_width=True
                        )
                    else:
                        st.warning("No objects detected in the image.")

                    # --- Download Button for Image ---
                    # Convert PIL Image to bytes for download
                    img_byte_arr = io.BytesIO()
                    result_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    st.download_button(
                        label="⬇️ Download Detected Image",
                        data=img_bytes,
                        file_name=f"detected_image_{model_choice.split()[0]}.png",
                        mime="image/png",
                        type="primary"
                    )

                except Exception as e:
                    st.error(f"An error occurred during image processing: {e}")

            elif input_type == 'video':
                temp_video_path = "temp_input_video." + file_extension
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                cap = cv2.VideoCapture(temp_video_path)
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                output_path = f"output_{model_choice.split()[0]}.mp4"
                
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                except:
                    st.warning("H.264 codec not found. Falling back to mp4v (video might not play in browser, but download will work).")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    results = model(frame, conf=confidence)
                    
                    res_plotted = results[0].plot()
                    
                    out.write(res_plotted)
                    
                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                out.release()
                progress_bar.empty()

                if os.path.exists(output_path):
                    st.header("🎥 Detection Result")
                    
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.video(video_bytes)

                    st.download_button(
                        label="⬇️ Download Detected Video (.mp4)",
                        data=video_bytes,
                        file_name=output_path,
                        mime="video/mp4",
                        type="primary"
                    )
                    
                    os.remove(temp_video_path)


st.markdown("---")
st.subheader("Model Class Information")

class_col1, class_col2 = st.columns(2)

with class_col1:
    st.markdown("##### ANPR Model (`yolo_ANPR.pt`)")
    st.markdown(f"**Number of Classes:** {len(ANPR_CLASSES)}")
    st.code(f"Classes: {ANPR_CLASSES}")

with class_col2:
    st.markdown("##### ATCC Model (`yolo_ATCC.pt`)")
    st.markdown(f"**Number of Classes:** {len(ATCC_CLASSES)}")
    st.code(f"Classes: {ATCC_CLASSES}")