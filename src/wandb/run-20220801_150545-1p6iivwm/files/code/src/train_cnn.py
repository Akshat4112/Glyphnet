from matplotlib import pyplot
from keras.utils import to_categorical
from keras import optimizers
from keras import layers
from keras import models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import tensorflow.keras.backend as K
import wandb
from wandb.keras import WandbCallback
import argparse
import os

wandb.init(project="HomoglyphDetection", entity="robofied")


print(device_lib.list_local_devices())
# print(K._get_available_gpus())

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg  = parser.parse_args().path_data

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

wandb.config = {"learning_rate": 1e-4,
                "epochs": 30,
                "batch_size": 1}

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# create data generator
datagen = ImageDataGenerator(rescale=1./255)

# prepare iterator
print("Before datagen..")
train_dir = os.path.join(path_arg, "final_train")
train_it = datagen.flow_from_directory(train_dir,
  class_mode='binary', batch_size=1, target_size=(256, 256))
print("Datagen completed..")

valid_dir = os.path.join(path_arg, "final_valid")
validation_it = datagen.flow_from_directory(valid_dir,
  class_mode='binary', batch_size=1, target_size=(256, 256))

# fit model
print("Training Starting..")
model.fit(train_it,validation_data=validation_it, steps_per_epoch=500, epochs=30, verbose=1, callbacks=[WandbCallback()])
print("Training Completed..")
# save model
print("Saving model to disk in models/")
model.save('../models/model_v1.h5')

print("Evalutaing model..")

# prepare iterator
print("Before datagen..")
test_dir = os.path.join(path_arg, "final_test")
test_it = datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=1, target_size=(256, 256))
print("Datagen completed..")

# evaluate model
print("Evaluating Model..")

_, acc = model.evaluate_generator(test_it, steps=500, verbose=0)
print('> %.3f' % (acc * 100.0))

print("Computing Precision and Recall for Classification.")

