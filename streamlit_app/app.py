from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import gdown
import tensorflow as tf
import streamlit as st
# App setups
###  Model URL and path
model_url = "https://drive.google.com/uc?id=1F3CBhna3a1y12JBtghrEWud4_bVJO8Kj"
model_path = "denseNet_v4.h5"
### Image folder path and class names
class_name=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
# Google Drive file IDs for each image
image_folder_path="https://drive.google.com/drive/folders/15eyz7Dyj3AddSQlXNLUu_tuvPiJZT5K3"
image_urls = {
    'CaS': '1q5Q5LcaAw3DNmRICS0Tx_tcjQaZAgbwN',
    'CoS': '1iktWqTizvTOAC083y_iORVvcFp3v9BM1',
    'Gum': '1OGFu_5D7afAn_KEXF93_ZxH_YaFkmpjB',
    'MC': '1hb6lMtqA9afGX7ncSYOzhZqNwBTg9pAJ',
    'OC': '1VOk9RK1YmMDoOecpGji69kt0IUZetbrs',
    'OLP': '1rjJz9UzxL6q-VeYG3KihXI1W071aWp5E',
    'OT': '1WyTOU5tiAb0F9GnyiP79O1jGMZ_l6VDf'
}
image_paths = {
    key: os.path.join('images', f'{key}.jpg') 
    for key in image_urls.keys()
}

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

## Caching the model download and load process
@st.cache_resource(ttl=24*3600)  # Cache for 24 hours
def get_model():
    
    """Downloads and loads model with caching"""
    # 1. Download (if needed)
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(model_url, model_path, quiet=True)
    
    # 2. Load model
    with st.spinner("Loading model..."):
        return load_model(model_path)
    
# Download images
def download_images():
    """Downloads example images if they don't exist"""
    for key, file_id in image_urls.items():
        if not os.path.exists(image_paths[key]):
            try:
                gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t", image_paths[key], quiet=False)
            except Exception as e:
                st.error(f"Failed to download {key} image: {e}")

# Initialize model and images
model = get_model()
#download_images()

# predict the class of the image Function
def predict(uploaded_file):
    image = Image.open(uploaded_file).resize((224, 224))
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    image=preprocess_input(image)
    return class_name[int(np.argmax(model.predict(image),axis=1))]

# Streamlit app
st.title("DentalNet : Intelligent Dental Classification with Deep Learning")
st.markdown("""
Welcome to **DentalNet**  an AI-powered dental image classification tool designed to assist in the early detection and categorization of common oral health conditions.  

Using deep learning and a custom-trained Convolutional Neural Network (CNN), this model can classify teeth into seven distinct categories:  
**OC (Occlusal Caries)**, **OT (Other Caries)**, **COS (Caries on Smooth Surface)**, **CAS (Caries on Approximal Surface)**, **MC (Missing Crown)**, **OLP (Overlapping Teeth)**, and **GUM (Gum Inflammation)**.  

Simply upload a dental image, and let the model analyze and predict the condition  helping streamline dental diagnostics with AI support.
""")
# st.subheader("Visual Examples: 7 Tooth Condition Categories")
# st.caption("Below are sample images from each of the seven classes used in our model.")

# cols = st.columns(3)

# for index, (key, value) in enumerate(image_paths.items()):
#     col = cols[index % 3]
#     with col:
#         st.image(value, caption=key, width=150)
       
       
uploaded_file = st.file_uploader("Upload a dental image for classification", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image",   width=200)
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction = predict(uploaded_file)
            st.success(f"The model predicts this image as: **{prediction}**") 