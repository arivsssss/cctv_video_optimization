import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import tempfile

# Load Keras model
model_path = "E:\cctv_camera\cctv_video_optimized_with_accuracy_0.8054.h5"
model = tf.keras.models.load_model(model_path)

def process_image(image, threshold):
    # Get model input shape from the model itself
    input_shape = model.input_shape[1:3]  # Get (height, width)
    
    # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    img = cv2.resize(img_rgb, input_shape)
    normalized_img = img / 255.0
    input_data = np.expand_dims(normalized_img, axis=0)  # Add batch dimension

    # Run inference
    pred = model.predict(input_data, verbose=0)[0][0]
    
    # Apply threshold
    label = "Human" if pred > threshold else "No Human"
    return label, pred, img

def process_video(video_path, threshold, num_frames=40):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    results = []
    for i in range(num_frames):
        frame_index = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            label, pred, processed_frame = process_image(frame, threshold)
            results.append((frame, processed_frame, label, pred))
    cap.release()
    return results

# Streamlit UI (remainder of the code stays the same)
st.title("Human Detection App")
threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.76, 0.01)

uploaded_file = st.file_uploader("Upload Video or Image", 
                               type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = np.array(Image.open(uploaded_file))
        label, pred, processed_img = process_image(image, threshold)
        
        st.image(image, caption=f"Original Image", use_column_width=True)
        st.image(processed_img, caption=f"Processed Image | {label} ({pred:.2f})", 
                use_column_width=True)
        
    else:
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        results = process_video(video_path, threshold)
        
        st.subheader("Video Analysis Results")
        cols = st.columns(4)
        for i, (orig_frame, proc_frame, label, pred) in enumerate(results):
            with cols[i % 4]:
                st.image(orig_frame, caption=f"Frame {i+1}: {label} ({pred:.2f})", 
                         use_column_width=True)
                
        human_count = sum(1 for r in results if r[2] == "Human")
        st.write(f"**Human detected in {human_count}/{len(results)} frames**")

st.write("Note: The model analyzes 40 evenly spaced frames from videos")