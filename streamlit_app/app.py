from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
# App setups
class_name=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
images={
    'CaS': 'images/a_100_0_1462.jpg',   
    'CoS': 'images/b_100.jpg',
    'Gum': 'images/g_1200.jpg',
    'MC': 'images/mc_1200.jpg',
    'OC': 'images/oc_1200.jpg',
    'OLP': 'images/p_1200.jpg',
    'OT': 'images/ot_1200.jpg'    
}
model = load_model('denseNet_v4.h5')
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
st.subheader("Visual Examples: 7 Tooth Condition Categories")
st.caption("Below are sample images from each of the seven classes used in our model.")
cols = st.columns(3)
for index, (key, value) in enumerate(images.items()):
    col = cols[index % 3]
    with col:
        st.image(value, caption=key, width=150)
       
       
uploaded_file = st.file_uploader("Upload a dental image for classification", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image",   width=200)
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction = predict(uploaded_file)
            st.success(f"The model predicts this image as: **{prediction}**") 