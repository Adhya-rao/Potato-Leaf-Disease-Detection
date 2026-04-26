# 🥔 Potato Leaf Disease Detection 🌿

This project is a **Deep Learning based Image Classification Web App** that detects **potato leaf diseases** using a **CNN (Convolutional Neural Network)** model.

It classifies images into:
- 🟢 Healthy
- 🟤 Early Blight
- ⚫ Late Blight

---

## 🚀 Features

- 🧠 CNN model for image classification
- 🖼️ Upload leaf image for prediction
- ⚡ Real-time prediction using Streamlit
- 🎯 Displays predicted disease class
- 📊 Trained on PlantVillage dataset

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- PIL (Image Processing)
- Matplotlib

---

## 📁 Project Structure

potato-disease-app/
│── app.py
│── cnn_model.keras
│── requirements.txt
│── README.md
---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
python -m venv venv

### 2️⃣ Activate Environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run Application
streamlit run app.py

---

## 🧠 Model Details

- Model Type: CNN (Convolutional Neural Network)
- Input Size: 224 × 224 × 3
- Layers:
  - Conv2D + MaxPooling (4 blocks)
  - Flatten
  - Dense Layer
  - Softmax Output (3 classes)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

---

## 📊 Dataset

Dataset: PlantVillage (Potato Leaves)

Classes:
- Potato___Healthy
- Potato___Early_blight
- Potato___Late_blight

---

## 🖥️ Application UI

### 🔹 Screenshot 1
<img width="451" height="701" alt="Screenshot 2026-04-26 232142" src="https://github.com/user-attachments/assets/671102f8-4aa9-4fc2-b9ed-9e852b616695" />



### 🔹 Screenshot 2
<img width="403" height="557" alt="Screenshot 2026-04-26 232312" src="https://github.com/user-attachments/assets/4b01bfa7-8571-49eb-8253-c5acb5531c31" />




### 🔹 Screenshot 3
<img width="431" height="606" alt="Screenshot 2026-04-26 232353" src="https://github.com/user-attachments/assets/5826eaca-796d-4813-b73d-7e5f2783af90" />




---

## 🧪 How It Works

1. Upload a potato leaf image  
2. Image is resized to 224×224  
3. Pixel values normalized (0–1)  
4. Passed into CNN model  
5. Model predicts disease class  

---

## ⚠️ Notes

- Ensure `cnn_model.h5` is present in project folder  
- Image should be clear for better prediction  
- Model accuracy depends on training quality  

---

## 📌 Future Improvements

- Use Transfer Learning (MobileNet / ResNet)  
- Improve accuracy with more data  
- Add confidence score display  
- Deploy using cloud platforms  

---

## 👩‍💻 Author

Adhya  

---

## ⭐ If you like this project, give it a star!
