

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='trained.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(imgp,model):
  IMG_SIZE=256
  path='/content'
  img_array = cv2.imread(os.path.join(path,imgp) ,cv2.IMREAD_GRAYSCALE)  # convert to array
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
  arr= np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  arr=arr/255.0
  pre=model.predict(arr)
  plt.barh(["Aluminium","Copper","Iron","Manganese"],pre[0], align='center', label="Data 1")
  plt.legend()
  plt.ylabel('Minerals')
  plt.xlabel('Probability of mineral presence')
  plt.title('Prediction')
  plt.show()
  pred=np.argmax(pre,axis=1)
  if pred[0]==0:
    preds='Aluminium is present.'
  if pred[0]==1:
    preds='Copper is present.'
  if pred[0]==2:
    preds='Iron is present.'
  if pred[0]==3:
    preds='Manganese is present.'
  return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('upload.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("I am running")
    if request.method == 'POST':
        print(request.files)
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
