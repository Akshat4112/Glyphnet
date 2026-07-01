import os
import argparse

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, roc_auc_score,
                             confusion_matrix)

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, default="../data", help="Define path for the data")
parser.add_argument("--model", type=str, default="../models/model_v1.h5", help="Path to the saved .h5 model")
args = parser.parse_args()
path_arg = args.path_data

model = load_model(args.model)

# Infer channel count from the model so this works with both the legacy
# 3-channel models and the current grayscale (1-channel) models.
channels = model.input_shape[-1]
color_mode = 'grayscale' if channels == 1 else 'rgb'

datagen = ImageDataGenerator(rescale=1./255)

print("Before datagen..")
test_dir = os.path.join(path_arg, "final_test")
# shuffle=False keeps predict() output aligned with test_it.classes
test_it = datagen.flow_from_directory(test_dir, class_mode='binary',
                                      color_mode=color_mode, batch_size=32,
                                      shuffle=False, target_size=(256, 256))
print("Datagen completed..")

predictions = model.predict(test_it).ravel()
y_true = test_it.classes
# Single sigmoid output -> threshold at 0.5
y_pred = (predictions > 0.5).astype(int)

print('Accuracy: %f' % accuracy_score(y_true, y_pred))
print('Precision: %f' % precision_score(y_true, y_pred))
print('Recall: %f' % recall_score(y_true, y_pred))
print('F1 score: %f' % f1_score(y_true, y_pred))
print('Cohens kappa: %f' % cohen_kappa_score(y_true, y_pred))
print('ROC AUC: %f' % roc_auc_score(y_true, predictions))
print('Confusion matrix:')
print(confusion_matrix(y_true, y_pred))
