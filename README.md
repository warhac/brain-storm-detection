# 🧠 BSD-Pro – Brain Scan Detector AI System

BSD-Pro is an AI-powered medical image classification system designed to detect brain tumors from MRI scan images using a Convolutional Neural Network (CNN).

---

## 🚀 Project Overview

This system provides:

- Tumor Type Prediction
- Confidence Score Display
- Probability Visualization (Graph)
- Automated Report Generation
- Modular and Scalable Architecture

---

## 🧠 Tumor Classes

The model classifies MRI scans into:

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## 🏗 Project Architecture

Brain-Tumor-Detection/
│
├── Training/ # Training Dataset
├── Testing/ # Testing Dataset
├── brain_tumor_model2.h5 # Trained CNN Model
│
├── core/ # AI Logic Layer
├── app/ # UI Layer
├── reports/ # Generated Reports
│
└── README.md

---

## 🔄 System Workflow

1. User uploads MRI image  
2. Image is preprocessed (resize + normalization)  
3. CNN model predicts tumor class  
4. Highest probability class is selected  
5. Confidence score is calculated  
6. Probability graph is displayed  
7. Report is generated and stored  

---

## 🧪 Model Information

- Model Type: Convolutional Neural Network (CNN)
- Input Size: 150x150 RGB
- Output Layer: Softmax
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

---

## 🛠 Installation

Install required libraries:

```bash
pip install tensorflow gradio opencv-python matplotlib reportlab pillow

python app/BSD.py
