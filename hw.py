
# Import Necessary Modules

import glob
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
# % matplotlib
# inline

image_dim = 64 * 64

def read_data():
    # get image filenames
    cat_locs = glob.glob('catsfolder/*.jpg')
    dog_locs = glob.glob('dogsfolder/*.jpg')
    num_cats = len(cat_locs)
    num_dogs = len(dog_locs)

    # initialize empty arrays
    X_cats = np.zeros((num_cats, 64 * 64))
    X_dogs = np.zeros((num_dogs, 64 * 64))
    y_cats = np.zeros((num_cats, 1))
    y_dogs = np.zeros((num_dogs, 1))

    # Load data, reshape into a 1D vector and set labels

    keep_track = 0

    for i in range(len(cat_locs)):
        img = cat_locs[i]
        im = io.imread(img)
        im = im.reshape(64 * 64)
        X_cats[i, :] = im
        y_cats[i] = -1.0
        keep_track += 1

    for i in range(len(dog_locs)):
        img = dog_locs[i]
        im = io.imread(img)
        im = im.reshape(64 * 64)
        X_dogs[i, :] = im
        y_dogs[i] = 1.0
        keep_track += 1

    # combine both datasets
    X = np.append(X_cats, X_dogs, 0)
    y = np.append(y_cats, y_dogs)

    return X, y

def show_image(X, i):
    # select image
    image = X[i, :]
    # reshape make into a square
    image = image.reshape((64, 64))
    # display the image
    plt.imshow(image, 'gray')

def calculate_accuracy(ytrue, yguess):
    # compare your predictions with the correct labels to determine how many of your predictions were correct.
    correct = sum((ytrue == yguess))
    total = len(ytrue)
    accuracy = 100 * float(correct) / float(total)
    # divide the number of correct predictions by the number of total samples to determine your classification accuracy.
    return accuracy

def split_data(X, y, testpercent):
    [n, d] = X.shape

    ntest = int(round(n * (float(testpercent) / 100)))
    ntrain = int(round(n - ntest))

    Xtrain = np.zeros((ntrain, d))
    Xtest = np.zeros((ntest, d))
    ytrain = np.zeros((ntrain, 1))
    ytest = np.zeros((ntest, 1))

    Data = np.column_stack((X, y))
    Data = np.random.permutation(Data)

    for i in range(ntest):
        Xtest[i, :] = Data[i, 0:d]
        ytest[i] = Data[i, d]

    for i in range(ntrain):
        Xtrain[i, :] = Data[i + ntest, 0:d]
        ytrain[i] = Data[i + ntest, d]

    return Xtrain, ytrain, Xtest, ytest

def pca(X):
    covX = np.cov(X, rowvar=False)
    [Lambda, Vtranspose] = np.linalg.eig(covX)
    neworder = np.argsort(-abs(Lambda))
    pcaX = Vtranspose[:, neworder]
    pcaX = pcaX.real
    return pcaX

def average_pet(X, y):
    # FILL IN CODE
    avgcat = [0] * image_dim
    avgdog = [0] * image_dim
    size_of_data = len(X)

    for i in range(0, size_of_data):
        for j in range(0, image_dim):
            if y[i] == 1:
                avgdog[j] += X[i][j]
            else:
                avgcat[j] += X[i][j]

    for i in range(0, image_dim):
        avgcat[i] = int(avgcat[i] / size_of_data)
        avgdog[i] = int(avgdog[i] / size_of_data)

    # render cat image to see
    cat_image = np.array(avgcat).reshape((64, 64))
    plt.imshow(cat_image, 'gray')

    # render dog image to see
    dog_image = np.array(avgdog).reshape((64, 64))
    plt.imshow(dog_image, 'gray')

    # return
    return avgcat, avgdog

def closest_average(Xtrain, ytrain, Xrun):
    # FILL IN CODE
    size_of_tests = len(Xrun)
    avgcat, avgdog = average_pet(Xtrain, ytrain)
    yguess = [0] * size_of_tests

    for i in range(0, size_of_tests):
        euclidean_cat = 0
        euclidean_dog = 0
        for j in range(0, image_dim):
            euclidean_cat += (Xrun[i][j] - avgcat[j]) ** 2
            euclidean_dog += (Xrun[i][j] - avgdog[j]) ** 2
        euclidean_cat = euclidean_cat ** (1 / 2)
        euclidean_dog = euclidean_dog ** (1 / 2)
        yguess[i] = -1 if euclidean_cat <= euclidean_dog else 1
    return yguess

def nearest_neighbor(Xtrain, ytrain, Xrun):
    # FILL IN CODE
    test_size = len(Xrun)
    yguess = [0] * test_size

    test_index = 1

    for i in range(0, test_size):
        nearest_neighbor_index = -1
        min_of_nearest = np.iinfo(np.int32).max
        curr_test = Xrun[i, :]
        for j in range(0, len(Xtrain)):
            euclidean = np.linalg.norm(Xtrain[j] - curr_test)
            if euclidean < min_of_nearest:
                min_of_nearest = euclidean
                nearest_neighbor_index = j

        yguess[i] = ytrain[nearest_neighbor_index, 0]
        # print(test_index, ": ", curr_test, " Result: ", yguess[i])
        test_index += 1
    return yguess





