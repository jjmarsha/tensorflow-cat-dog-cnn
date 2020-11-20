from tensorflow import keras as kerast
from skimage import io
from hw import read_data, split_data, calculate_accuracy
import sys
import glob
import numpy as np

def pet_classifier(X):
    model = kerast.models.load_model("pet_classifier_trainedModel")
    X = X.reshape(len(X),64,64,1)
    prediction = model.predict_classes(X)

    yguess = np.zeros((2000, 1))
    for i, value in enumerate(prediction):
        yguess[i] = -1 if value == 0 else 1
    return yguess[:,0]
