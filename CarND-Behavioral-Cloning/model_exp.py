import scipy.misc
import scipy.stats
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Lambda, ELU, Dense, Dropout, Activation, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt

train_offset = 0
val_offset = 0
nvidia_dims = (66, 200)

def load_nvidia_data():
    # center images
    xs = []
    # steering angles
    ys = []
    with open("./driving_dataset/data.txt") as f:
        for line in f:
            xs.append("./driving_dataset/" + line.split()[0])
            # nivida data uses inverse of turing radius 
            ys.append(float(line.split()[1]) * scipy.pi / 180)
    return xs, ys

def brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def flip(image, steering):
    return cv2.flip(image,1), -steering

def shear(image, steering, shear_range=100):
    rows, cols, ch = image.shape
    numPixels = 10
    steeringShiftPerPixel = 0.4
    transX = shear_range * np.random.uniform() - shear_range/2
    steering = steering + transX/shear_range * 2 * steeringShiftPerPixel
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1,0, transX], [0,1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering

def load_udacity_data():
    # CSV layout: center,left,right,steering,throttle,brake,speed
    # images
    xs = []
    # steering angles
    ys = []
    with open("./data/driving_log.csv") as f:
        for line in f:
            if 'IMG' in line:
                xs.append(line)
                steering_degrees = float(line.split(",")[3])
                ys.append(steering_degrees)           
    return xs, ys

def load_self_captured_data():
    # CSV layout: center,left,right,steering,throttle,brake,speed
    # images
    xs = []
    # steering angles
    ys = []
    with open("./self_data/driving_log.csv") as f:
        for line in f:
            if 'IMG' in line:
                xs.append(line)
                steering_degrees = float(line.split(",")[3])
                ys.append(steering_degrees)           
    return xs, ys

def split(X, y, test_pct, valid_pct):
    # first split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct, random_state=832289)
    # next split training to training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
        test_size=valid_pct, random_state=832289)
    print('Training set: ', len(y_train), ' Validation set: ', len(y_val), ' Test set: ', len(y_test))
    return X_train, y_train, X_val, y_val, X_test, y_test

def batch_prepare(X, y, offset, batch_size, dims, train):
    left_right_correction_degrees = 0.25
    X_batch = []
    y_batch = []
    count = len(X)
    for i in range(batch_size):
        line = X[offset % count]
        if "Users" in line:
            file_tag = ""
        else:
            file_tag = "./data/"
        if train:
            steering = y[i]
            camera = np.random.choice(['center', 'left', 'right'])
            if camera == 'center':
                img = scipy.misc.imread(file_tag + (line.split(",")[0]).strip())
            if camera == 'left':
                img = scipy.misc.imread(file_tag + (line.split(",")[1]).strip())
                steering += left_right_correction_degrees
            if camera == 'right':
                img = scipy.misc.imread(file_tag + (line.split(",")[2]).strip())
                steering -= left_right_correction_degrees

            # reshape from 160x320 to 76x280
            img = img[42:118,20:300,:]
            img = scipy.misc.imresize(img, dims)

            img, steering = brightness(img), steering
            img, steering = shear(img, steering, 100)
            if np.random.rand() > 0.5:
                img, steering = flip(img, steering)
        else:
            img = scipy.misc.imread(file_tag + (line.split(",")[0]).strip())
            steering = y[i]

            # reshape from 160x320 to 76x280
            img = img[42:118,20:300,:]
            img = scipy.misc.imresize(img, dims)

        X_batch.append(img)
        y_batch.append(steering)
    return np.array(X_batch), np.array(y_batch)

def plot_class_distribution(labels, info):
    print('Info: ', info)
    hist, bins = np.histogram(labels, bins=np.arange(-0.5, 0.5, 0.001))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
def nvidia_model(dims):
    in_shape = (dims[0], dims[1], 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=in_shape, output_shape=in_shape))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse", metrics=['mean_absolute_error'])
    return model
    
def train_generator(training_set, training_labels, batch_size, dims, train):
    global train_offset
    while True:
        X, y = batch_prepare(training_set, training_labels, train_offset, batch_size, dims, train)
        train_offset += batch_size
        yield X, y 

def train_with_generator(model, training_set, training_labels,validation_set, validation_labels, 
    batch_size, epochs, train, dims):
    history = model.fit_generator(generator=train_generator(training_set, training_labels, batch_size, dims, train), 
                                  samples_per_epoch=6000, nb_epoch=epochs,
                                  verbose=1, 
                                  validation_data=(validation_set, validation_labels))
    
def test(model, test_set, test_labels):
    score = model.evaluate(test_set, test_labels, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

def save(model):
    json_string = model.to_json()
    file = open("model.json", "w")
    file.write(json_string)
    file.close()
    model.save_weights("model.h5")
  
def load_data():
    return load_udacity_data()
 
def process():
    batch_size = 300
    epochs = 10
    dims = nvidia_dims
    X, y = load_data()
    # plot_class_distribution(y, 'Steering Angle Distribution')
    X_train, y_train, X_val, y_val, X_test, y_test = split(X, y, 0.2, 0.05)
    model = nvidia_model(dims)
    X_val, y_val = batch_prepare(X_val, y_val, 0, len(X_val), dims, False)
    train_with_generator(model, X_train, y_train, X_val, y_val, batch_size, epochs, True, dims)
    X_test, y_test = batch_prepare(X_test, y_test, 0, len(X_test), dims, False)
    test(model, X_test, y_test)
    save(model)
    
process()

