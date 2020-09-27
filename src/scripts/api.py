from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import urllib.request
import pickle
import os
import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow.keras

app = Flask(__name__)

print("model Loaded")
font = 'ARIAL.TTF'

# # BASE_DIR = ''

def img(text, path):     
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill = (255))
    #enhancer = ImageEnhance.Contrast(img)
    #im_output = enhancer.enhance(1.5)
    #transposed  = img.transpose(Image.ROTATE_90)
    img.save(path + text + '.png', 'PNG')
    return path+text+".png"
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(256, 256))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 256, 256, 3)
	# center pixel data
	img = img.astype('float32')
	return img

# load an image and predict the class
def get_pred(image_path):
  # load the image
  img = load_image(image_path)
  # load model
  model = load_model('../../models/model_v1.h5')
  # predict the class
  result = model.predict(img)
#  print(result[0])
  return result


def take_input(text):
    image_path = img(text, '../../data/all_classes/')
    result = get_pred(image_path)
    print("This image is: ", result)
    return result[0]

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    domain = req_data['domain']
    result = take_input(domain)
    if result[0] == 1.0:
        prediction = "real"
    elif result[0] == 0.0:
        prediction = "fake"
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



#take input(domain_name)


#if st.button("Predict"):
 #   result = take_input(domain_name)
  #  print("Result is: ", result)
   # if result[0] == 1.0:
    #    st.success("Real Domain")
   # elif result[0] == 0.0:
    #    st.error("Fake Domain")

