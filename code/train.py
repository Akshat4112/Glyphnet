from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
import tensorflow.keras.backend as K
from tensorflow.keras import Input
from tensorflow.keras import Model
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

import argparse
import os
from datetime import datetime
from attentionModule import attach_attention_module
import matplotlib.pyplot as plt

class NeuralNetwork:
  def __init__(self) -> None:
    wandb.init(project="HomoglyphDetection", entity="robofied")
    print(device_lib.list_local_devices())
    # print(K._get_available_gpus())
    pass

  def DataGenerator(self, train_dir, valid_dir):
    self.train_dir = train_dir
    self.valid_dir = valid_dir

    # create data generator
    datagen = ImageDataGenerator(rescale=1./255)

    # prepare iterator
    print("Before datagen..")
    self.train_it = datagen.flow_from_directory(train_dir,class_mode='binary', batch_size=1, target_size=(256, 256))
    print("Datagen completed..")
    self.validation_it = datagen.flow_from_directory(valid_dir,class_mode='binary', batch_size=1, target_size=(256, 256))
    
  def SimpleCNN(self):

    wandb.config = {"learning_rate": 1e-4,
                    "epochs": 30,
                    "batch_size": 1}

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

    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    self.history = model.fit(self.train_it,validation_data=self.validation_it, steps_per_epoch=50, epochs=5, verbose=1, callbacks=[WandbCallback(), callback])
    
    # save model
    self.name = '../models/modelSimpleCNN' + str(datetime.now()) + '.h5'
    model.save(self.name)

  def AttentionCNN(self):
    inputs = Input(shape=(256, 256, 3))

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = attach_attention_module(x, attention_module='cbam_block')

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = attach_attention_module(x, attention_module='cbam_block')

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = attach_attention_module(x, attention_module='cbam_block')

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D(2, 2)(x)
    x = attach_attention_module(x, attention_module='cbam_block')
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    self.history = model.fit(self.train_it,validation_data=self.validation_it, steps_per_epoch=500, epochs=5, verbose=1, callbacks=[WandbCallback(), callback])
    
    # save model
    self.name = '../models/modelAttentionCNN' + str(datetime.now()) + '.h5'
    model.save(self.name)

    return None

  def plotGraphs(self):
    history = self.history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    acc_fig_name = '../figures/' + '_acc_' + self.name[10:-1] + '.png'
    plt.savefig(acc_fig_name)
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    loss_fig_name = acc_fig_name = '../figures/' + '_loss_' + self.name[10:-1] + '.png'
    plt.savefig(loss_fig_name)
    
  # def Evaluation():
  #       # fit model

  #   print("Evalutaing model..")

  #   # prepare iterator
  #   print("Before datagen..")
  #   test_dir = os.path.join(path_arg, "final_test")
  #   test_it = datagen.flow_from_directory(test_dir, class_mode='binary', batch_size=1, target_size=(256, 256))
  #   print("Datagen completed..")

  #   # evaluate model
  #   print("Evaluating Model..")

  #   _, acc, f1_score, precision, recall = model.evaluate_generator(test_it, steps=500, verbose=0)
  #   print('> %.3f' % (acc * 100.0))

  #   print("Computing Precision and Recall for Classification.")

  #   return None

obj = NeuralNetwork()
obj.DataGenerator('../data/train', '../data/valid')
print("Data Generation Completed")
obj.SimpleCNN()
obj.plotGraphs()