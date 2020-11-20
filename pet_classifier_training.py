import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras as keras2
from hw import read_data, split_data
import numpy as np

batch_size = 64
num_classes = 2
epochs = 20

col_dim = 64
row_dim = 64
image_dim = col_dim * row_dim

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

X, y = read_data()
for i, value in enumerate(y):
    y[i] = 0 if value == -1 else 1
x_train, y_train, x_test, y_test = split_data(X, y, 20)

x_train = x_train.reshape(1600,row_dim,col_dim,1)
x_test = x_test.reshape(400,row_dim,col_dim,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# block 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(row_dim,col_dim,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.20))

# block 2
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.10))

# block 3
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

optimizer_cat_dog = keras2.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save("./pet_classifier_trainedModel")
print('Test loss:', score[0])
print('Test accuracy:', score[1])