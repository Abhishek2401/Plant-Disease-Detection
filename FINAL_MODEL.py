#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
try:  
    from PIL import Image
except ImportError:  
    import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2
import requests
from io import BytesIO
warnings.filterwarnings('ignore')
filename = 'model.hdf5'
classifier = tf.keras.models.load_model('model.hdf5')
df = pd.read_csv('Diseases.csv')
li = []
for i in range(len(df)):
    li.append(df['Diseases'][i])
    
def plant(image_path):  
    #Preprocessing image
    new_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    
    #Predicting the diseases
    prediction = classifier.predict(img)
    
    batch_size = 128
    d = prediction.flatten()
    j = d.max()
    for index,item in enumerate(d):
    	if item == j:
    		class_name = li[index]
    return class_name


# In[3]:


import os  
from flask import Flask, render_template, request


# define a folder to store and later serve the images
UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','jfif'])

app = Flask(__name__)

# function to check the file extension
def allowed_file(filename):  
    return '.' in filename and            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# route and function to handle the home page
@app.route('/')
def home_page():  
    return render_template('index.html')

# route and function to handle the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    global disease
    disease = ''
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('upload.html', msg='No file selected')
        
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))

        if file and allowed_file(file.filename):

            
            img_src=UPLOAD_FOLDER + file.filename
            disease = plant(img_src)

           
            return render_template('upload.html',
                                   msg='Successfully processed:\n'+file.filename,
                                   disease=disease,
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('upload.html')
    
@app.route('/upload/symptoms', methods=['GET', 'POST'])
def symptoms():
    try:
        if disease is not None:
            for i in range(len(df)):
                if(disease == df['Diseases'][i]):
                    symptoms = df['Symptoms'][i]
            return render_template('symptoms.html',symptoms = symptoms)
    except:
        return render_template('symptoms.html',symptoms = "No file selected")

@app.route('/upload/treatement', methods=['GET', 'POST'])
def treatment():
    try:
        if disease is not None:
            for i in range(len(df)):
                if(disease == df['Diseases'][i]):
                    treatment = df['Treatment'][i]
            return render_template('treatment.html',treatment = treatment)
    except:
        return render_template('treatment.html',treatment = "No file selected")
if __name__ == '__main__':  
    app.run(debug = False)