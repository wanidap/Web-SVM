import os
import uuid
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import  InceptionV3,DenseNet201
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input


def title():
    # Declare title app
    title_container = st.container()
    col1, col2 = st.columns([20,1])
    image = Image.open("image_logo.png")
    with title_container:
        with col1:
            st.markdown('<h1 style="color: blue;">Pneumonia Classification Web App</h1>',
                            unsafe_allow_html=True)
        with col2:
            st.image(image, width=200)
    #st.title("Pneumonia Classification Web App")
    st.write("""Predicting Pneumonia Using Stochastic Sub-gradient Support Vector Machine with generalized Pinball loss function (SGD-GSVM) from Chest X-ray Images.""")
    st.subheader('Demo for image classification.')
    img = Image.open("Pneumonia.png")
    st.image(img, width=500)


def get_feature_(img):
    # load base model
    base_model = DenseNet201(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.reshape(1920,)
    return features

def predict(features):
    w_b = pd.read_csv('w_b.csv').values[:,-1]
    p = np.sign(np.matmul(features,w_b[:-1])+w_b[-1])
    return p

def main():
    # set title
    title()

    # enable users to upload images for the model to make predictions
    file_up = st.file_uploader("Upload an image", type=['png','jpeg','jpg'])

    if file_up is not None:
        #random_name = str(uuid.uuid4()) # get unique name for the uploaded img
        #path = os.path.join(".", "userUpload", f"{random_name}.jpg") # get the saved img's path
        #Image.open(file_up).save(path) # save the img to the disk
        #st.image(Image.open(path), caption = 'Uploaded Image.', width=200) # display image that user uploaded
        st.image(file_up, caption = 'Uploaded Image.', width=200)
        st.write("")
        st.write("Just a second ...")

        # get a feature vector
        #img = image.load_img(path, target_size=(224, 224)) # load the saved img from the path and convert to (299,299,3) RGB img
        img = Image.open(file_up)
        img = img.convert('RGB')
        img = img.resize((224,224))

        feature = get_feature_(img) # get feature vector from CNN model

        # prediction step
        p = predict(feature) # predict the class by using the CNN feature of the img.
        if p >= 0:
            st.write("""Prediction: **Normal**""")
        else:
            st.write("""Prediction: **Pneumonia**""")


if __name__=="__main__":
    main()
