import os
import uuid
from PIL import Image, ImageOps

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import  InceptionV3,DenseNet201
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input


def title():
    # Declare title app
    #title_container = st.container()
    #col1, col2 = st.columns([20,1])
    #image = Image.open("logo1.png")
    #with title_container:
        #with col1:
           # st.markdown('<h1 style="color: navy;">Pneumonia Classification</h1>',
                         #   unsafe_allow_html=True)
        #with col2:
            #st.image(image, width=200)
    #st.title("Pneumonia Classification Web App")
    st.markdown(f'<h1 style="color:navy;font-size:45px;">{"Pneumonia Classification Web App"waving_black_flag"}</h1>', unsafe_allow_html=True)
    st.write("""Predicting Pneumonia Using Stochastic Sub-gradient Support Vector Machine with generalized Pinball loss function (SGD-GSVM) from Chest X-ray Images. Pneumonia is the most common disease caused
by various microbial species such as bacteria, viruses, and fungi that inflame the air
sacs in one or both lungs. There are 5,856 x-ray images out of which 4,273 are positive for Pneumonia infection i.e. Pneumonia (+) and the rest 1,583 are negative for Pneumonia infection i.e. Normal (-).""")
    #st.subheader('Demo for image classification.')
    #col1, col2 = st.columns(2)
    #original = Image.open("IM-0007-0001.jpeg")
    #col1.subheader("Normal")
    #col1.image(original, use_column_width=True)

    #grayscale = Image.open("person5_bacteria_15.jpeg")
    #col2.subheader("Pneumonia")
    #col2.image(grayscale, use_column_width=True)
    img = Image.open("Pneumonia.png")
    st.image(img, width=700)
    st.write("For the Pneumonia recognition dataset, we resized all images to an appropriate size based on the CNN model, and every image had been converted from RGB to grayscale color image. Moreover, feature extraction is a powerful technology that influences image recognition accuracy. We have used an automatic extraction algorithm called Convolutional Neural Networks (CNN) for feature extraction.")
    st.write("The training and testing accuracy for Pneumonia recognition dataset are 94.96% and 94.11%, respectively which is given by DenseNet201 CNN combined with our SGD-GSVM model.")
    with st.sidebar:
        st.markdown(f'<h1 style="color:navy;font-size:25px;">{"Contact:"}</h1>', unsafe_allow_html=True)
        #st.header("Contact:")
        st.write("Wanida Panup")
        st.write("Tel.  (+66)8 8436 5416")
        st.write("Email. wanidap56@nu.ac.th")
        st.write("Address: 54/1, Moo 4, Khun Fang subdistrict, Mueang Uttaradit, Uttaradit Province, Thailand 53000")
        img = Image.open("logo1.png")
        st.image(img, width=300)
        col1, col2 = st.columns(2)
        original = Image.open("NUlogo.png")
        col1.image(original, use_column_width=True)

        grayscale = Image.open("SCIlogo.png")
        col2.image(grayscale, use_column_width=True)
        #img = Image.open("NUlogo.png")
        #st.image(img, width=100)
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
        st.image(file_up, caption = 'Uploaded Image.', width=300)
        st.write("")
        st.write("Just a second ...")

        img = Image.open(file_up)
        img = img.convert('RGB')
        img = img.resize((224,224))

        feature = get_feature_(img) # get feature vector from CNN model

        # prediction step
        p = predict(feature) # predict the class by using the CNN feature of the img.
        if p >= 0:
            st.write("""**Prediction**: _Normal_.""")
        else:
            st.write("""**Prediction**: _Pneumonia_.""")


if __name__=="__main__":
    main()
