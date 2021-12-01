import streamlit as st
import heapq
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image,ImageOps
from tensorflow.keras.preprocessing.image import img_to_array


a_file = open("Streamlit/labels.pkl", "rb")
output = pickle.load(a_file)


labels = list(output.values())

def make_predictions(model,data,labels):
    predictions = model.predict(data)
    predictions = np.resize(predictions,(predictions.shape[1],))
    classes = heapq.nlargest(3, range(len(predictions)), predictions.take)
    lab = [labels[i] for i in classes]
    return lab,predictions[classes]

st.title("Flower Classification")
st.sidebar.title("Created By:")
st.sidebar.markdown("Kavya S Kumar")
st.sidebar.subheader("Github account: ")
st.sidebar.markdown("https://github.com/kavyasureshkumar")
st.sidebar.header("Models trained on the dataset")
col1,col2 = st.sidebar.columns(2)
col1.markdown("### Model Name")
col1.write("EffecientNet")
col1.write("MobileNet")
col2.markdown("### Accuracy")
col2.write("70%")
col2.write("66%")
image = st.file_uploader("Choose an image", type = "jpeg")
if image is None:
    st.write("Please select an image")
else:
    image1 = Image.open(image)
    st.image(image1,width = 300, caption='Uploaded Image.')
    option = st.selectbox("Choose a model ",("EffecientNet","MobileNet"))
    with st.spinner("Getting the model ready..."):
        if option == "EffecientNet":
            model = tf.keras.models.load_model('Streamlit/model/EffecientNet.h5')
        else:
            model = tf.keras.models.load_model('Streamlit/model/mobile_net.h5')
    predict = st.button("Predict")
    if predict:
        with st.spinner('Predicting ...'):
            image_ = np.array(image1)
            image_ = np.resize(image1,(1,224,224,3))
            image_ = image_/255
            lab,prob = make_predictions(model,image_,labels)
            st.header("The uploaded image is")
            col1,col2,col3 = st.columns(3)
            col1.header(lab[0])
            col1.subheader(str(round(prob[0]*100,2)) + "% sure")
            col2.header(lab[1])
            col2.subheader(str(round(prob[1]*100,2)) + "% sure")
            col3.header(lab[2])
            col3.subheader(str(round(prob[2]*100,2)) + "% sure")
            st.success("The prediction is complete!")







