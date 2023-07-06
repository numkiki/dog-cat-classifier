from keras.models import load_model  
import tensorflow as tf
from PIL import Image, ImageOps  
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt

def predict(img):
    # Read the model
    model_name = 'keras_model.h5'
    size = (224, 224)
    
    # Load the model
    model = load_model(model_name, compile=False)

    # Read the labels.txt to initialize the classification
    class_names = open("labels.txt", "r").readlines()

    # Change the image to 224x224 according to the Teachable Machine's input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Normalize the image based on the variable data
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the data
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]

    # Return the label
    return class_name[2:]

# Title of the Streamlit
st.title('Dog and Cat classifier')

# Create a file uploader
file_upload = st.file_uploader("Choose file", 
                               type = ['png', 'jpg', 'jpeg'])

# Create a button to perform classification
class_btn = st.button("Classify!")

# Condition when uploading a file
if file_upload is not None:    
    image = Image.open(file_upload)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Condition when using the button
if class_btn:
    if file_upload is None:
        st.write("Invalid command, please upload an image")
    else:
        with st.spinner('Model working....'):
             plt.imshow(image)
             plt.axis("off")
             predictions = predict(image)
             time.sleep(1)
             st.success('Classified')
             st.write(predictions)