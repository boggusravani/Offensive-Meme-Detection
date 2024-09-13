from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import pickle
import nltk
import joblib

import os
import easyocr
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, render_template, request, redirect, Response

import pandas as pd
import numpy as np

app = Flask(__name__)

cv = pickle.load(open("model.pkl", 'rb'))

model = joblib.load('model.sav')

reader = easyocr.Reader(['en'])

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    
    print("Entered here")
    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")

    result = reader.readtext(file_path,paragraph="False")
    var1 = result[0][1] 
    try:
        var2 = result[1][1]
        print(var2)
    except IndexError:
        var2 = '!'
        print(var2)

    var3 = " ".join([var1, var2])
    data = [var3]
    vect = cv.transform(data).toarray()
    my_prediction = model.predict(vect)


    #pred, output_page = model_predict1(file_path,model)
              
    return render_template('result.html', prediction  = my_prediction, img_src=UPLOAD_FOLDER + file.filename)


   
if __name__ == '__main__':
    app.run(debug=False)