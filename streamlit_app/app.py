import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import os
import pandas as pd
import tempfile

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 300px; # Adjust this value as needed (e.g., 250px or 350px)
        max-width: 350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Try importing fast_plate_ocr
try:
    from fast_plate_ocr import LicensePlateRecognizer
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# --- Configuration ---
MODEL_OPTIONS = {
    "ANPR (Automatic Number Plate Recognition)": "yolo_ANPR.pt",
    "ATCC (Traffic Count & Classification)": "yolo_ATCC.pt",
}

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Vision: ANPR & Traffic Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Fluid UI ---
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Caching Resources ---
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_ocr_model():
    if OCR_AVAILABLE:
        # Initializing the global model as requested
        return LicensePlateRecognizer('cct-s-v1-global-model')
    return None

# --- Helper Functions ---
def draw_ocr_text(image, text, box, color=(0, 255, 0)):
    """Draws text and bounding box on a PIL image."""
    draw = ImageDraw.Draw(image)
    # Draw Box
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw Text Background and Text
    # Load a font (try default or fallback)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    draw.rectangle([x1, y1 - 25, x2, y1], fill=color)
    draw.text((x1 + 5, y1 - 25), text, fill="white", font=font)
    return image

def run_ocr_on_crop(image_np, box, ocr_model):
    """
    Crops the image, saves it as a temp file, runs OCR, and cleans up.
    """
    if ocr_model is None:
        return "OCR_LIB_MISSING"
    
    # 1. Coordinate handling
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = image_np.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        return None

    # 2. Crop the image
    crop = image_np[y1:y2, x1:x2]
    
    # 3. Save to a temporary file
    try:
        # Create a temp file but don't delete immediately (we need to read it first)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            # Save the crop
            Image.fromarray(crop).save(temp_path)
        
        # 4. Pass the FILE PATH to the model
        print(f"DEBUG: Running OCR on {temp_path}")
        prediction = ocr_model.run(temp_path)
        
        # 5. Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return prediction, crop

    except Exception as e:
        print(f"OCR Error: {e}")
        # Ensure cleanup happens even if OCR fails
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None
# --- Main App Logic ---

def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        selected_task = st.selectbox("Select Task / Model", list(MODEL_OPTIONS.keys()))
        model_path = MODEL_OPTIONS[selected_task]
        
        is_anpr = "ANPR" in selected_task
        
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload Media", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        
        run_btn = st.button("🚀 Run Analysis", type="primary")

        st.info("Note: Ensure 'fast_plate_ocr' is installed for License Plate reading.")

    # --- Main Content ---
    st.title("👁️ AI Vision: Object & Plate Detection")

    if not uploaded_file:
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 50px;'>
            <h3>👋 Welcome!</h3>
            <p>Please upload an Image or Video in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Load Models
    yolo_model = load_yolo_model(model_path)
    ocr_model = load_ocr_model() if is_anpr else None
    
    if is_anpr and not OCR_AVAILABLE:
        st.error("❌ `fast_plate_ocr` library not found. ANPR will detect plates but cannot read numbers.")

    # Determine File Type
    file_type = uploaded_file.name.split('.')[-1].lower()
    is_video = file_type in ['mp4', 'avi', 'mov', 'mkv']
    
    # --- Processing ---
    if run_btn:
        st.divider()
        
        if not is_video:
            process_image_mode(uploaded_file, yolo_model, ocr_model, is_anpr, conf_threshold)
        else:
            process_video_mode(uploaded_file, yolo_model, ocr_model, is_anpr, conf_threshold)

def process_image_mode(uploaded_file, model, ocr_model, is_anpr, conf):
    col_img, col_data = st.columns([3, 2])
    
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    with st.spinner("Analyzing Image..."):
        results = model.predict(image, conf=conf)
        detections = results[0].boxes
        
        detected_data = []
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for box in detections:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            display_text = label
            
            # Logic for ANPR
            if is_anpr and label == "license_plate":
                plate_text, _ = run_ocr_on_crop(image_np, xyxy, ocr_model)
                if plate_text:
                    display_text = f"{plate_text}"
                    detected_data.append({"Object": label, "Details": plate_text, "Confidence": float(box.conf[0])})
            else:
                detected_data.append({"Object": label, "Details": "N/A", "Confidence": float(box.conf[0])})

            # Draw Custom Boxes/Text
            # (We use PIL drawing here to avoid Canvas limitations and have cleaner text)
            if is_anpr:
                annotated_image = draw_ocr_text(annotated_image, display_text, xyxy)
            else:
                # Standard bounding box for ATCC
                draw.rectangle(xyxy, outline="red", width=3)
                draw.text((xyxy[0], xyxy[1]-10), display_text, fill="red")

    # --- Result Display ---
    with col_img:
        st.subheader("🖼️ Processed Image")
        st.image(annotated_image, use_container_width=True)
        
        # Download Image
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        st.download_button("⬇️ Download Result Image", data=buf.getvalue(), file_name="result.png", mime="image/png")

    with col_data:
        st.subheader("📊 Detection Data")
        if detected_data:
            df = pd.DataFrame(detected_data)
            st.dataframe(df, use_container_width=True)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV Report",
                csv,
                "detection_report.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("No objects detected.")


    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Setup
    output_path = "output_video.mp4"
    # Try H264 first, fallback to mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detection_log = []
    
    st.write(f"**Processing Video:** {total_frames} frames at {fps:.2f} FPS")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Current timestamp
        current_time_sec = frame_count / fps
        
        # Detection
        results = model(frame, conf=conf, verbose=False)
        boxes = results[0].boxes

        # Frame Data Store
        frame_detections = []

        # Convert to RGB for PIL drawing (Better text rendering)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        if len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # Logic for ANPR vs ATCC
                if is_anpr and label == "license_plate":
                    # OCR
                    plate_text = run_ocr_on_crop(frame, xyxy, ocr_model)
                    
                    if plate_text:
                        # Add to log
                        detection_log.append({
                            "Timestamp (s)": round(current_time_sec, 2),
                            "Frame": frame_count,
                            "License Plate": plate_text,
                            "Confidence": float(box.conf[0])
                        })
                        
                        # Draw on frame (Live Caption Effect)
                        # We draw a large "HUD" style text above the car
                        draw.rectangle(xyxy, outline=(0, 255, 0), width=3)
                        
                        # Background for text
                        font = ImageFont.load_default() # Or load ttf if available
                        text_disp = f"PLATE: {plate_text}"
                        text_bbox = draw.textbbox((xyxy[0], xyxy[1]), text_disp, font=font)
                        draw.rectangle([xyxy[0], xyxy[1]-20, xyxy[2], xyxy[1]], fill=(0, 255, 0))
                        draw.text((xyxy[0]+5, xyxy[1]-20), text_disp, fill="black", font=font)
                
                elif not is_anpr:
                    # ATCC Logic (Counting/Classifying)
                    detection_log.append({
                        "Timestamp (s)": round(current_time_sec, 2),
                        "Frame": frame_count,
                        "Type": label
                    })
                    # Draw standard box
                    draw.rectangle(xyxy, outline="blue", width=2)
                    draw.text((xyxy[0], xyxy[1]-10), label, fill="blue")

        # Convert back to BGR for VideoWriter
        final_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        # Add a global timestamp overlay on the video itself
        cv2.putText(final_frame, f"Time: {current_time_sec:.2f}s", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(final_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Processing Frame {frame_count}/{total_frames}")

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    os.remove(video_path) # Clean up input temp

    # --- Display Results ---
    st.success("Processing Complete!")
    
    tab1, tab2 = st.tabs(["🎥 Video Result", "💾 Data Logs"])
    
    with tab1:
        st.video(output_path)
        
        with open(output_path, 'rb') as v:
            st.download_button("⬇️ Download Processed Video", v, file_name="anpr_output.mp4")

    with tab2:
        st.subheader("Detailed Detection Log")
        if detection_log:
            df = pd.DataFrame(detection_log)
            
            # Logic to group unique plates if needed, but client asked for per-frame updates
            # displaying the raw log allows them to see exactly what was detected at 0.3s vs 0.6s
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV Log",
                csv,
                "video_detection_log.csv",
                "text/csv"
            )
        else:
            st.warning("No detections found in video.")

def process_video_mode(uploaded_file, model, ocr_model, is_anpr, conf):

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())

    tfile.close() 
    
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    # Video Properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Setup
    output_path = "output_video.mp4"
    
    # Try H264 (avc1) first, fallback to mp4v if it fails
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # FIX: Check if writer actually opened. If not, force fallback.
    if not out.isOpened():
        print("DEBUG: H.264 (avc1) codec failed. Falling back to mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detection_log = []
    
    st.write(f"**Processing Video:** {total_frames} frames at {fps:.2f} FPS")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Sidebar Live View
    with st.sidebar:
        st.subheader("🚗 Live Plate View")
        live_plate_spot = st.empty()
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time_sec = frame_count / fps
        
        # Detection
        results = model(frame, conf=conf, verbose=False)
        boxes = results[0].boxes

        # Convert to RGB for PIL drawing
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        if len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                if is_anpr and label == "license_plate":
                    # Run OCR
                    plate_text, crop_img = run_ocr_on_crop(frame, xyxy, ocr_model)
                    
                    if plate_text:
                        plate_text_str = str(plate_text) # Ensure string
                        
                        detection_log.append({
                            "Timestamp (s)": round(current_time_sec, 2),
                            "Frame": frame_count,
                            "License Plate": plate_text_str,
                            "Confidence": float(box.conf[0])
                        })
                        
                        # Live Sidebar Update
                        with live_plate_spot.container():
                            st.image(crop_img, caption=f"Plate: {plate_text_str}", width=200)

                        # Draw HUD on frame
                        draw.rectangle(xyxy, outline=(0, 255, 0), width=3)
                        font = ImageFont.load_default()
                        text_disp = f"PLATE: {plate_text_str}"
                        text_bbox = draw.textbbox((xyxy[0], xyxy[1]), text_disp, font=font)
                        draw.rectangle([xyxy[0], xyxy[1]-20, xyxy[2], xyxy[1]], fill=(0, 255, 0))
                        draw.text((xyxy[0]+5, xyxy[1]-20), text_disp, fill="black", font=font)
                
                elif not is_anpr:
                    # ATCC
                    detection_log.append({
                        "Timestamp (s)": round(current_time_sec, 2),
                        "Frame": frame_count,
                        "Type": label
                    })
                    draw.rectangle(xyxy, outline="blue", width=2)
                    draw.text((xyxy[0], xyxy[1]-10), label, fill="blue")

        # Write Frame
        final_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        cv2.putText(final_frame, f"Time: {current_time_sec:.2f}s", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(final_frame)
        
        frame_count += 1
        if frame_count % 10 == 0 and total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Processing Frame {frame_count}/{total_frames}")

    # Cleanup Resources
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    # Safe File Removal
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except PermissionError:
            print(f"DEBUG: Could not delete temp file {video_path} (it might still be in use). Ignoring.")
            pass

    # --- Display Results ---
    st.success("Processing Complete!")
    
    tab1, tab2 = st.tabs(["🎥 Video Result", "💾 Data Logs"])
    
    with tab1:
        # Note: 'mp4v' videos might not play in all browsers, but download works.
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        
        with open(output_path, 'rb') as v:
            st.download_button("⬇️ Download Processed Video", v, file_name="anpr_output.mp4")

    with tab2:
        st.subheader("Detailed Detection Log")
        if detection_log:
            df = pd.DataFrame(detection_log)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV Log",
                csv,
                "video_detection_log.csv",
                "text/csv"
            )
        else:
            st.warning("No detections found in video.")

if __name__ == "__main__":
    main()