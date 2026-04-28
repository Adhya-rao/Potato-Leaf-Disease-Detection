import streamlit as st
import numpy as np
import pickle
from PIL import Image
import keras
import tensorflow as tf

st.set_page_config(page_title="Potato Disease Detection", layout="centered")
st.title("🥔 Potato Leaf Disease Detection")

@st.cache_resource
def load_files():
    try:
        # Match Colab's Keras 3 behavior
        model = keras.models.load_model("cnn_model.keras", compile=False)

        with open("class_indices.pkl", "rb") as f:
            class_indices = pickle.load(f)

        return model, class_indices, None
    except Exception as e:
        return None, None, str(e)

model, class_indices, error = load_files()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()
else:
    st.success("✅ Model Loaded Successfully (Keras 3.13.2)")

labels = {v: k for k, v in class_indices.items()}

uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess to match training
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype('float32')

    # Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions)
    label = labels[predicted_class]
    
    if "healthy" in label.lower():
        st.success(f"🌿 Prediction: {label}")
    else:
        st.error(f"⚠️ Prediction: {label}")
        
    st.write(f"**Confidence:** {np.max(predictions):.2%}")
