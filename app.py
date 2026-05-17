import streamlit as st
import numpy as np
import pickle
from PIL import Image
import keras
import tensorflow as tf

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Potato Disease Detection",
    layout="centered"
)

st.title("🥔 Potato Leaf Disease Detection")

st.info(
    "Supported Classes: Healthy, Early Blight, Late Blight"
)

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_files():

    try:

        model = keras.models.load_model(
            "cnn_model.keras",
            compile=False
        )

        with open("class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        return model, class_indices, None

    except Exception as e:

        return None, None, str(e)


model, class_indices, error = load_files()

# ---------------- ERROR HANDLING ---------------- #

if error:

    st.error(f"❌ Error loading model: {error}")
    st.stop()

else:

    st.success("✅ Model Loaded Successfully")


# ---------------- LABELS ---------------- #

labels = {v: k for k, v in class_indices.items()}

# ---------------- FILE UPLOAD ---------------- #

uploaded_file = st.file_uploader(
    "Upload ONLY Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ---------------- #

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

    # Convert image to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype('float32')

    # Prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = np.argmax(predictions)

    label = labels[predicted_class]

    # Output
    if "healthy" in label.lower():

        st.success(
            f"🌿 Prediction: {label}"
        )

    else:

        st.error(
            f"⚠️ Prediction: {label}"
        )

    # Confidence
    st.write(
        f"**Confidence:** {np.max(predictions):.2%}"
    )

    # Debug (optional)
    # st.write(predictions)
