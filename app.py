import streamlit as st
import numpy as np
import pickle
from PIL import Image
import keras
import tensorflow as tf
import os

# Page Configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    layout="centered"
)

st.title("🥔 Potato Leaf Disease Detection")



# Load Model and Labels
@st.cache_resource
def load_files():
    try:
        # Check if model file exists
        if not os.path.exists("cnn_model.keras"):
            return None, None, "cnn_model.keras file not found"

        # Load CNN model
        model = keras.models.load_model(
            "cnn_model.keras",
            compile=False
        )

        # Load class labels
        with open("class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        return model, class_indices, None

    except Exception as e:
        return None, None, str(e)

# Load resources
model, class_indices, error = load_files()

# Error handling
if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()

else:
    st.success("✅ Model Loaded Successfully")

# Reverse mapping
labels = {v: k for k, v in class_indices.items()}

# Upload image
uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype("float32")

    # Prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Confidence
    confidence = np.max(predictions)

    # Predicted class
    predicted_class = np.argmax(predictions)

    label = labels[predicted_class]

    # Validation
    if confidence < 0.70:

        st.warning(
            "⚠️ Please upload a valid potato leaf image"
        )

    else:

        # Healthy
        if "healthy" in label.lower():

            st.success(
                f"🌿 Prediction: {label}"
            )

        # Diseased
        else:

            st.error(
                f"⚠️ Prediction: {label}"
            )

        # Confidence score
        st.write(
            f"**Confidence:** {confidence:.2%}"
        )
