from tensorflow import keras as kerast
from skimage import io
from hw import read_data, split_data, calculate_accuracy
from pet_classifier import pet_classifier

X, y = read_data()
X = X.reshape(2000,64,64,1)
yguess = pet_classifier(X)
accuracy = calculate_accuracy(y, yguess)
print("Accuracy: {}%".format(accuracy))

