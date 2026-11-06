
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]["index"]
output_details = interpreter.get_output_details()[0]["index"]

st.title("face mask detection")

file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
if file is not None:
    image = tf.keras.utils.load_img(file)
    st.image(image, caption="Uploaded image")
    img = tf.image.resize(image, (256, 256))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    labels = ["with_mask", "witout_mask"]
    if st.button("Predict"):
        interpreter.set_tensor(input_details, img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details)
        st.write(labels[np.argmax(predictions)])



