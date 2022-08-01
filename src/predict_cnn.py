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
# from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import tensorflow.keras

#model = load_model('../../models/30epochs_model.h5')
#print("model Loaded")
font = '../data/ARIAL.TTF'

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
  model = load_model('../models/model_v1.h5')
  # predict the class
  result = model.predict(img)
  print(result[0])
  return result[0]


def take_input(text):
    image_path = img(text, '../data/all_classes/')
    result = get_pred(image_path)
    print("This image is: ", result)
    print(type(result[0]))
    if result[0]==0.0:
        print("It is real")
    return result[0]

print(take_input('www.google.com'))

