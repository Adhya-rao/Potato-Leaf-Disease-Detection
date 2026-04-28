# streamlit run app.py
import os
# Force legacy Keras behavior before importing TensorFlow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import numpy as np
import pickle
from PIL import Image
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
        model = tf.keras.models.load_model("cnn_model.keras")  # or cnn_model.h5

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
    image = np.expand_dims(image, axis=0)
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

    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)

    # Prediction
    predictions = model.predict(processed)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    label = labels[predicted_class]

    # -------------------------------
    # Display Result
    # -------------------------------
    if "healthy" in label.lower():
        st.success(f"🌿 Prediction: {label}")
    else:
        st.error(f"⚠️ Prediction: {label}")

    st.write(f"Confidence: {confidence:.4f}")

# -------------------------------
# Info
# -------------------------------
st.markdown("---")
st.write("Upload a potato leaf image to detect disease type.")
