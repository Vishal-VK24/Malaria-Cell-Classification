# -*- coding: utf-8 -*-


import sys
import os
import glob
import re
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
from keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras import layers, models
import tensorflow as tf

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

class_names=['gametocyte','leukocyte','red blood cell','ring','schizont','trophozoite']
from tensorflow.keras.models import load_model

app = Flask(__name__)

model= load_model("C:\Project\models\malaria__Mobile_net.h5")

#model._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    image=cv2.imread(img_path)
    image=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{img_path}1.png",image)
    image = tf.keras.preprocessing.image.load_img(f"{img_path}1.png")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    k=np.argmax(predictions)
    return k

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)

        result = str(class_names[preds])               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
