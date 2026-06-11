# 🥔 Potato Leaf Disease Detection using Deep Learning

## 📌 Project Overview

This project is a **Deep Learning-based web application** that helps farmers and agricultural users identify diseases in potato leaves from images.

The application classifies uploaded potato leaf images into one of the following categories:

* ✅ Healthy
* 🍂 Early Blight
* ⚠️ Late Blight

The main objective of this project is to provide a quick and accessible solution for early disease detection, helping reduce crop loss and improve agricultural productivity.

---

## 🚀 Features

* Upload potato leaf images through a simple web interface
* Automatic image preprocessing
* Disease prediction using a trained CNN model
* Instant display of classification results
* User-friendly Streamlit application

---

## 🏗️ Project Workflow

1. User uploads an image of a potato leaf.
2. The image undergoes preprocessing:

   * Resizing to **224 × 224 pixels**
   * Normalization of pixel values
3. The processed image is passed to a trained **Convolutional Neural Network (CNN)**.
4. The model extracts important visual features and predicts the disease category.
5. The prediction result is displayed instantly on the Streamlit interface.

---

## 🧠 Model Architecture

The project uses a **Convolutional Neural Network (CNN)** for image classification.

CNNs are highly effective for computer vision tasks because they automatically learn:

* Texture patterns
* Shape features
* Disease-specific visual characteristics

This makes CNNs well-suited for plant disease detection applications.

---

## 📊 Dataset

The model was trained using the **PlantVillage Dataset**, which contains labeled images of healthy and diseased potato leaves.

Classes used:

* Potato Healthy
* Potato Early Blight
* Potato Late Blight

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Web Framework

* Streamlit

### Libraries

* NumPy
* Pandas
* Pillow (PIL)
* Matplotlib

---

## 📂 Project Structure

```text
Potato-Leaf-Disease-Detection/
│
├── app.py
├── model/
│   └── trained_model.h5
│
├── dataset/
│
├── requirements.txt
│
├── README.md
│
└── assets/




### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---



## 🔍 Challenges Faced

### 1. Consistent Predictions

Different users upload images with varying sizes and quality.

**Solution:**

* Implemented image resizing
* Applied normalization
* Standardized input format before prediction

### 2. Deployment Issues

Deploying machine learning applications requires proper dependency management.

**Solution:**

* Configured `requirements.txt`
* Used Streamlit Cloud for seamless deployment

---

## 🎯 Results

The trained CNN model successfully classifies potato leaf images into:

* Healthy
* Early Blight
* Late Blight

The system provides fast and user-friendly disease detection through a web interface.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning
* Computer Vision
* Convolutional Neural Networks (CNNs)
* Image Processing
* Model Deployment
* Streamlit Web Applications
* End-to-End AI Development

---




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
## Deployed Link

   https://potato-leaf-disease-detection-m9dzmzchkfqd6mn3wde5xw.streamlit.app/

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
- app.py
- cnn_model.keras
- requirements.txt
- README.md

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
