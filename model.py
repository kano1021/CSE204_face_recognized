import sys

import random

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
 
from load_data import load_from, resize, resize_capture, IMAGE_SIZE

class Dataset:
    def __init__(self, path):
        # training set
        self.train_images = None
        self.train_labels = None
        
        # testing set
        self.test_images  = None            
        self.test_labels  = None
        
        # path of the data
        self.path = path

        self.classes = 0

    # Load data and preparation
    def load(self, type, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3): # type = 0 -> age; type = 1 -> gender; type = 2 -> ethnicity
        # Load data into memory
        images, a_labels, g_labels, e_labels = load_from(self.path)        
        
        if type == 0:
            self.classes = 10
            train_images, test_images, train_labels, test_labels = train_test_split(images, a_labels, test_size = 0.3, random_state = random.randint(0, 100))
        elif type == 1:
            self.classes = 2
            train_images, test_images, train_labels, test_labels = train_test_split(images, g_labels, test_size = 0.3, random_state = random.randint(0, 100))
        elif type == 2:
            self.classes = 5
            train_images, test_images, train_labels, test_labels = train_test_split(images, e_labels, test_size = 0.3, random_state = random.randint(0, 100))

        # Shape: number, rows, cols, channels
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_channels, img_rows, img_cols)                  
            
        # Output the shape of the training set and test set
        print(train_images.shape, 'train samples')
        print(test_images.shape, 'test samples')

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images  = test_images
        self.test_labels  = test_labels
        
        # Change the data into 2D vector.
        train_labels = to_categorical(train_labels, self.classes)                                   
        test_labels = to_categorical(test_labels, self.classes)                        
    
        # Regularization of the pixels
        train_images = train_images.astype('float32')/255
        test_images = test_images.astype('float32')/255

        print(train_images.shape, 'train samples')
        print(test_images.shape, 'test samples')

class Gender_Model:
    def __init__(self, dataset):
        self.model = Sequential()
        self.dataset = dataset
        self.trained = None

        #1 2D convolution layer
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        self.model.add(Activation('relu'))

        #2 Maxpooling layer
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        #3 2D convolution layer
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        #4 2D convolution layer
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        #5 Maxpooling layer
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))

        #6 Flatten layer
        self.model.add(Flatten())
        #7 Dense layer
        self.model.add(Dense(512))
        self.model.add(Activation('sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        #8 Output layer
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))

        # Print summary of the model
        # In this model we use binary_crossentropy
        #sgd = SGD(lr=0.05, decay=1.0e-6)
        self.model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        self.model.summary()
        return

    def train(self, batch_size=128, epochs=5, file_name=None):
        """
        Trains the model on the given data.
        """
        ds = self.dataset
        stop_early = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, mode='auto')
        checkpoint = ModelCheckpoint(file_name,monitor='val_acc',verbose=1,save_best_only=True, period=1)
        self.trained = self.model.fit(ds.train_images, ds.train_labels, batch_size=batch_size, epochs=epochs, validation_data=(ds.test_images, ds.test_labels), shuffle=True, callbacks=[stop_early, checkpoint])

        if file_name:
            self.model.save(file_name)
    
    def load_model(self, file_name):
        self.model = load_model(file_name)

    def predict(self, img):
        img = resize_capture(img)
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE, 3): return None
        img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        
        pred = self.model.predict_proba(img)
        print(pred)
        if pred[0][0] >= pred[0][1]:
            return 0
        else: return 1


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        ds = Dataset("/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/UTKFace")
        ds.load(1)
        gm = Gender_Model(ds)
        gm.load_model(file_name = '/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/gender_classifier.h5')
        gm.train(file_name="/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/gender_classifier.h5")
        #model = load_model("gender_classifier.h5")
        #score = model.evaluate(ds.test_images, ds.test_labels, verbose = 1)
        #print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))