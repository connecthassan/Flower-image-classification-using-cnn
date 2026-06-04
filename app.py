import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Page configuration
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌸 Flower Image Classification")
st.markdown("""
Welcome to the Flower Classification App! Upload an image of a flower to identify its species
using a deep learning CNN model.
""")

# Sidebar configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Minimum confidence level for predictions"
)

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Trained Model", "MobileNetV2 Transfer Learning"]
)

# Load model
@st.cache_resource
def load_model(model_type="trained"):
    """Load the trained model"""
    try:
        if model_type == "trained":
            # Try to load the trained model
            model_path = "models/flower_classification_model.h5"
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                return model
            else:
                st.warning("Trained model not found. Using placeholder.")
                return None
        else:
            # Load MobileNetV2 pretrained model
            model = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet'
            )
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Flower classes
FLOWER_CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_flower(image, model):
    """Make prediction on the flower image"""
    try:
        if model is None:
            return None, None
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return FLOWER_CLASSES[predicted_class_idx], confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a flower image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG, JPEG, or PNG image of a flower"
    )

with col2:
    st.subheader("Sample Images")
    if st.button("📸 Load Sample Image"):
        sample_path = "data/sample_flower.jpg"
        if os.path.exists(sample_path):
            uploaded_file = sample_path
            st.success("Sample image loaded!")
        else:
            st.info("No sample image available")

# Process uploaded image
if uploaded_file is not None:
    try:
        # Load and display image
        if isinstance(uploaded_file, str):
            image = Image.open(uploaded_file)
        else:
            image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)
        
        # Load model
        model = load_model(model_option)
        
        # Make prediction
        if model is not None:
            with st.spinner("🤔 Analyzing image..."):
                flower_class, confidence = predict_flower(image, model)
            
            with col2:
                st.subheader("Prediction Result")
                
                if flower_class is not None:
                    if confidence >= confidence_threshold:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="color: #2E7D32; text-align: center;">✓ {flower_class}</h2>
                            <p style="text-align: center; font-size: 18px;">
                                Confidence: <strong>{confidence:.2%}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence for all classes
                        st.subheader("Class Probabilities")
                        predictions = model.predict(preprocess_image(image), verbose=0)[0]
                        
                        for idx, class_name in enumerate(FLOWER_CLASSES):
                            col_a, col_b = st.columns([2, 3])
                            with col_a:
                                st.write(f"**{class_name}**")
                            with col_b:
                                st.progress(float(predictions[idx]))
                                st.write(f"{predictions[idx]:.2%}")
                    else:
                        st.warning(
                            f"⚠️ Low confidence prediction: {flower_class} ({confidence:.2%})\n\n"
                            f"Confidence below threshold ({confidence_threshold:.0%})"
                        )
                else:
                    st.error("Could not make a prediction")
        else:
            st.error("Model not loaded. Please check the model path or try a different model option.")
    
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Information section
st.markdown("---")
st.subheader("ℹ️ About")
st.markdown("""
**Supported Flower Classes:**
- 🌼 Daisy
- 🌻 Dandelion  
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip

**How to use:**
1. Upload an image of a flower
2. The model will analyze the image
3. Get instant predictions with confidence scores

**Model Info:**
- Architecture: Convolutional Neural Network (CNN)
- Input Size: 224x224 pixels
- Classes: 5 flower types
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌸 Flower Classification using CNN | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
