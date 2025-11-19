import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os 
import pandas as pd 
import time

# --- FIXES ---
# 1. FORCE CPU USAGE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# 2. Hardcode the overall model accuracy (Update this after your training run!)
MODEL_ACCURACY = 0.85 # Placeholder for a better trained model accuracy

# --- CUSTOM CSS FOR THEMED BACKGROUND ---
# Using a light recycling green background
THEME_COLOR = "#e9f5e9" 
BACKGROUND_CSS = f"""
<style>
.stApp {{
    background-color: {THEME_COLOR};
}}
/* Apply a slight shadow/border to the main content block for separation */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}
</style>
"""
st.markdown(BACKGROUND_CSS, unsafe_allow_html=True)
# ----------------------------------------


# --- 1. CONFIGURATION AND LOADING ---

# Load the saved model (Streamlit caches this for fast loading)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_classifier_resnet50.h5')

# Load class indices (to map prediction index back to class name)
@st.cache_resource
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class, class_to_idx

# Load the resources
idx_to_class, class_to_idx = load_class_indices()
model = load_model()
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- 2. PREDICTION FUNCTION ---

def predict_image(img):
    # FIX: ENSURE 3 CHANNELS (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_resized = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Simulate a longer prediction time for the interactive progress bar
    time.sleep(0.5) 
    
    # Make prediction
    predictions = model.predict(img_array)
    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)
    predicted_class = idx_to_class.get(predicted_index, "Unknown Class")
    confidence = probabilities[predicted_index]
    
    return predicted_class, confidence, probabilities

# --- 3. STREAMLIT FRONT-END LAYOUT (THEMED) ---

st.set_page_config(page_title="SDG 12 Waste Classifier", layout="wide")

# Main Header
st.title("♻️ AI-Powered Waste Sorting (SDG 12)")
st.caption("Using ResNet50 for 30-class waste item classification.")

# --- Controls Row ---
control_col1, control_col2 = st.columns([1, 2])

with control_col1:
    # Display Overall Model Accuracy
    st.metric(
        label="Overall Model Accuracy", 
        value=f"{MODEL_ACCURACY*100:.1f}%", 
        help="Accuracy measured on the validation set during training."
    )

with control_col2:
    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload a waste item image (.jpg, .png) here to begin analysis:", type=["jpg", "jpeg", "png"])
    st.info("Tip: The model works best with well-centered images of single waste items.")


st.markdown("---")

if uploaded_file is not None:
    
    # Use columns to display image and prediction side-by-side
    image_col, result_col = st.columns([1, 2])
    
    # Display the uploaded image
    with image_col:
        img = Image.open(uploaded_file)
        st.subheader("🖼️ Uploaded Item")
        # Use a border for the image for a cleaner look
        st.image(img, use_column_width=True)
    
    # Prediction logic
    with result_col:
        st.subheader("🔬 Analysis & Results")
        
        # Add an interactive button
        if st.button('🚀 Start Classification', type="primary", use_container_width=True):
            
            # Use an interactive progress bar
            progress_bar = st.progress(0, text="Starting analysis...")
            
            # Simulate progress
            for percent_complete in range(100):
                if percent_complete < 30:
                    text = "Preprocessing image and normalizing..."
                elif percent_complete < 90:
                    text = "Running ResNet50 prediction on CPU..."
                else:
                    text = "Finalizing prediction results..."
                    
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1, text=text)

            class_name, confidence, probabilities = predict_image(img)
            
            # --- INTERACTIVE EFFECT: CONFETTI ---
            st.balloons() # Or st.snow() or st.confetti()
            # ------------------------------------

            # Clear the progress bar
            progress_bar.empty()
            st.success('✅ Classification Complete!')
            
            # Display primary results with a large header
            st.markdown(f"## **Predicted Waste Type:** {class_name}")
            
            # Use a colored box for the confidence score
            st.markdown(f"**Confidence Score:** <span style='font-size: 1.5em; color: green;'>**{confidence*100:.2f}%**</span>", unsafe_allow_html=True)

            # --- Interactive Plot and Details ---
            st.markdown("#### Top 10 Probability Distribution")

            # Create a dataframe for plotting
            class_names = [idx_to_class[i] for i in range(len(probabilities))]
            df = pd.DataFrame({
                'Class': class_names,
                'Probability': probabilities
            }).sort_values(by='Probability', ascending=False)
            
            # Display Top 10 probabilities as a chart
            st.bar_chart(
                data=df.head(10), 
                x='Class', 
                y='Probability', 
                use_container_width=True
            )
            
            # Use an expander for detailed data (more interactive)
            with st.expander("⬇️ Show All 30 Class Probabilities (Full Data Table)"):
                st.dataframe(df, use_container_width=True)