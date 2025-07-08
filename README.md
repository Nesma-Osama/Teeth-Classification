# 🦷 Teeth Classification with CNN and DenseNet169

This project focuses on classifying dental images into 7 distinct categories using two approaches:

- A custom-built **Convolutional Neural Network (CNN)**
- A **pretrained DenseNet169 model** fine-tuned on the dataset

The model is deployed via a user-friendly [Streamlit_app link](https://nesma-osama-teeth-classification-streamlit-appapp-yza35y.streamlit.app/) app where users can upload dental images for real-time classification.

---
## 🔍 Model Overview

### 1. 🧠 CNN Model
- A custom 6-layer CNN architecture trained from scratch.
- Pooling and activation layers are not counted toward the "6 layers" as they contain no trainable parameters, following standard research conventions .
- Used data augmentation to enhance generalization.
- Achieved strong performance with relatively low complexity.

### 2. 🧠 DenseNet169 (Pretrained)
- Used `DenseNet169` from Keras Applications with `imagenet` weights.
- Removed the top layer and added:
  - GlobalAveragePooling2D
  - Dense layers with ReLU
  - Dropout for regularization
  - Final Dense layer with 7-class softmax output
- Fine-tuned the last 50 layers for better feature extraction.

---

## 📊 Dataset

The dataset includes **7 classes** of dental conditions or types:

OC

OT

COS

CAS

MC

OLP

GUM

## 🚀How to Run the Streamlit App Locally
1- Clone the Repository

2- Go to the streamlit_app Folder

3- Install the Requirements
  
    ```pip install -r requirements.txt```
    
4- Run the Streamlit App

     ```streamlit run app.py```
