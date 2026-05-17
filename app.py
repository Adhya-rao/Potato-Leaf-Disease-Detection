import streamlit as st
import numpy as np
import pickle
from PIL import Image
from tensorflow import keras

# Page config
st.set_page_config(
    page_title="Potato Disease Detection",
    layout="centered"
)

# Title
st.title("🥔 Potato Leaf Disease Detection")

# Load model and labels
@st.cache_resource
def load_files():
    try:
        # Load trained model
        model = keras.models.load_model(
            "cnn_model.keras",
            compile=False
        )

        # Load class labels
        with open("class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        return model, class_indices

    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return None, None


# Load resources
model, class_indices = load_files()

# Stop if loading failed
if model is None or class_indices is None:
    st.stop()

st.success("✅ Model Loaded Successfully")

# Reverse dictionary
labels = {v: k for k, v in class_indices.items()}

# File uploader
uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# Prediction section
if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype("float32")

    # Predict
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Get prediction
    predicted_class = np.argmax(predictions)

    # Get label
    label = labels[predicted_class]

    # Confidence
    confidence = np.max(predictions)

    # Display result
    st.subheader("Prediction Result")

    if "healthy" in label.lower():
        st.success(f"🌿 {label}")
    else:
        st.error(f"⚠️ {label}")

    st.write(f"### Confidence: {confidence:.2%}")
