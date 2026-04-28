# streamlit run app.py
import os
# Force legacy Keras behavior before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pickle
from PIL import Image
import keras  # We use direct keras for .keras files
import tensorflow as tf

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Potato Disease Detection", layout="centered")

st.title("🥔 Potato Leaf Disease Detection")

# -------------------------------
# Load Model & Classes
# -------------------------------
@st.cache_resource
def load_files():
    try:
        # Load using keras directly with compile=False to avoid version conflicts
        model = keras.models.load_model("cnn_model.keras", compile=False)

        with open("class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        return model, class_indices, None

    except Exception as e:
        return None, None, str(e)

model, class_indices, error = load_files()

# -------------------------------
# Error Handling
# -------------------------------
if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Tip: Ensure your requirements.txt has tensorflow>=2.16 and keras>=3.0")
    st.stop()
else:
    st.success("✅ Model Loaded Successfully")

# Reverse mapping
labels = {v: k for k, v in class_indices.items()}

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype('float32')
    return image

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        processed = preprocess_image(image)
        # Prediction
        predictions = model.predict(processed, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        label = labels[predicted_class]

        # -------------------------------
        # Display Result
        # -------------------------------
        st.subheader("Result:")
        if "healthy" in label.lower():
            st.success(f"🌿 Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.write(f"**Confidence:** {confidence:.2%}")

# -------------------------------
# Info
# -------------------------------
st.markdown("---")
st.caption("Potato Disease Classifier | Powered by TensorFlow & Streamlit")

