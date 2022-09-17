import random
import numpy as np
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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


#Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(hash("setting random seeds") % 2**32 -1)
np.random.seed(hash("improves reproducibility")% 2**32 -1)
tf.random.set_seed(hash("by removing stochasticity")% 2**32 -1)


# print(device_lib.list_local_devices())

class NeuralNetwork:
  def __init__(self) -> None:
    pass

  def DataGenerator(self, train_dir, valid_dir):
    self.train_dir = train_dir
    self.valid_dir = valid_dir

    # create data generator
    self.datagen = ImageDataGenerator(rescale=1./255)

    # prepare iterator
    print("Before datagen..")
    self.train_it = self.datagen.flow_from_directory(train_dir,class_mode='binary', batch_size=1, target_size=(256, 256))
    print("Datagen completed..")
    self.validation_it = self.datagen.flow_from_directory(valid_dir,class_mode='binary', batch_size=1, target_size=(256, 256))
    
  def SimpleCNN(self, config):
    self.config = config
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 3)))
    self.model.add(layers.MaxPool2D(2, 2))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    self.model.add(layers.MaxPool2D(2, 2))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    self.model.add(layers.MaxPool2D(2, 2))
    self.model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    self.model.add(layers.MaxPool2D(2, 2))
    self.model.add(layers.Flatten())
    self.model.add(layers.Dense(128, activation='relu'))
    self.model.add(layers.Dense(1, activation='sigmoid'))
    self.model.summary()
    self.model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=self.config['learning_rate']),
                  metrics=['acc'])
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    WandbCallback(monitor="vall_loss", save_mode=(True), log_weights=(True))

    self.history = self.model.fit(self.train_it,
                                  validation_data=self.validation_it, 
                                  steps_per_epoch=config['steps_per_epoch'], 
                                  epochs=self.config['epochs'], 
                                  batch_size=self.config['batch_size'], 
                                  verbose=1, 
                                  callbacks=[WandbCallback()])
    
    # save model
    self.name = '../models/modelSimpleCNN' + str(datetime.now()) + '.h5'
    self.model.save(self.name)

  def AttentionCNN(self, config):
    self.config = config

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
    self.model = Model(inputs, outputs)
    
    self.model.summary()
    self.model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=self.config['learning_rate']),
                  metrics=['acc'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    WandbCallback(monitor="vall_loss", save_mode=(True), log_weights=(True))

    self.history = self.model.fit(self.train_it,
                                  validation_data=self.validation_it, 
                                  steps_per_epoch=config['steps_per_epoch'], 
                                  epochs=self.config['epochs'], 
                                  batch_size=self.config['batch_size'], 
                                  verbose=1, 
                                  callbacks=[WandbCallback()])

    # save model
    self.name = '../models/modelAttentionCNN' + str(datetime.now()) + '.h5'
    self.model.save(self.name)


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
    
  def Evaluation(self):
    # create data generator
    print("Evalutaing model..")
    self.test_it = self.datagen.flow_from_directory("../data/test", class_mode='binary', batch_size=1, target_size=(256, 256))
    
    train_loss, train_acc = self.model.evaluate(self.train_it, steps=10, verbose=0)
    test_loss, test_acc = self.model.evaluate(self.test_it, steps=10, verbose=0)
    
    print('Training Loss is: ', train_loss)
    print('Training Accuracy is: %.3f' % (train_acc * 100.0))

    print('Test Loss is: ', test_loss)
    print('Test Accuracy is: %.3f' % (test_acc * 100.0))

    true_labels = self.test_it.classes
    predictions = self.model.predict(self.test_it)
    y_true = true_labels
    y_pred = np.array([np.argmax(x) for x in predictions])
    
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(y_true, y_pred)
    print('Precision: %f' % precision)
    recall = recall_score(y_true, y_pred)
    print('Recall: %f' % recall)
    f1 = f1_score(y_true, y_pred)
    print('F1 score: %f' % f1)
    kappa = cohen_kappa_score(y_true, y_pred)
    print('Cohens kappa: %f' % kappa)
    auc = roc_auc_score(y_true, predictions)
    print('ROC AUC: %f' % auc)
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)

    wandb.log({'Accuracy': accuracy, 
              'Precision': precision, 
              'Recall': recall, 
              'f1-score': f1, 
              'kappa': kappa, 
              'auc': auc, 
              'confusion_matrix': matrix})


run = wandb.init(project="HomoglyphDetection", entity="robofied")


obj = NeuralNetwork()

obj.DataGenerator('../data/train', '../data/valid')
print("Data Generation Completed")

config = wandb.config = {"learning_rate": 1e-4,
                          "epochs": 3,
                          "steps_per_epoch":50,
                          "batch_size": 16,
                          "architecture":"Simple CNN",
                          "dataset":"Glyphnet Dataset",}

obj.SimpleCNN(config)
obj.plotGraphs()
obj.Evaluation()
print("Simple CNN Experiment Completed")
run.finish()



run = wandb.init(reinit=True)

config = wandb.config = {"learning_rate": 1e-4,
                          "epochs": 3,
                          "steps_per_epoch":50,
                          "batch_size": 16,
                          "architecture":"Attention CNN",
                          "dataset":"Glyphnet Dataset",}


obj.AttentionCNN(config)
obj.plotGraphs()
obj.Evaluation()
wandb.finish()

print("Attention CNN Experiment Completed")