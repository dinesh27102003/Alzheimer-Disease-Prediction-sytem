import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt  
sns.set_theme(style="darkgrid")
sns.set()

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'C:/Users/preethi/Documents/alzheimers_cnn_svm_rf/my_model2.hdf5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Alzheimer Disease Prediction
         """)

file = st.file_uploader("Please upload a brain scan file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (128, 128)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert("L")  # Convert to grayscale
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ..., np.newaxis]  # Add single channel

    # Ensure that the input image has the correct shape and data type
    img_reshape = img_reshape.astype('float32') / 255.0

    # Make predictions using the loaded model
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['non demented', 'mild demented', 'demented', 'very mild demented']
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    
    string = f"This person is : {class_names[predicted_class]} "
    st.success(string)
    

