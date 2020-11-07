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
        #训练集
        self.train_images = None
        self.train_labels = None
        
        #测试集
        self.test_images  = None
        self.test_labels  = None
        
        #数据集加载路径
        self.path = path
        
        self.classes = 0
    
    #加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, type, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3): # type = 0 -> age; type = 1 -> gender; type = 2 -> ethnicity
        #加载数据集到内存
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

        #shape: number, rows, cols, channels
        train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
        test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
        self.input_shape = (img_channels, img_rows, img_cols)
        
        #输出训练集、验证集、测试集的数量
        print(train_images.shape, 'train samples')
        print(test_images.shape, 'test samples')

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images  = test_images
        self.test_labels  = test_labels
        
        #我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        #类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_labels = to_categorical(train_labels, self.classes)
        test_labels = to_categorical(test_labels, self.classes)
        
        #像素数据浮点化以便归一化;将其归一化,图像的各像素值归一化到0~1区间
        train_images = train_images.astype('float32')/255
        test_images = test_images.astype('float32')/255

        print(train_images.shape, 'train samples')
        print(test_images.shape, 'test samples')

class Ethni_Model:
    def __init__(self,dataset):
        self.model = Sequential()
        self.dataset = dataset
        self.trained = None
        
        #1 2维卷积层
        self.model.add(Conv2D(32, (3, 3), padding='same',data_format="channels_first", input_shape=( IMAGE_SIZE, IMAGE_SIZE,3)))
        self.model.add(Activation('relu'))
        #2 2维卷积层
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        #3 最大池化层
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        
        
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        
        
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        
        

        #7 Flatten layer
        self.model.add(Flatten())
        #8 Dense layer
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        
        # Output layer
        self.model.add(Dense(5))
        self.model.add(Activation('sigmoid'))
        
        #输出模型概况
        #sgd = SGD(lr=0.1, decay=1.0e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.model.summary()
    


    def train(self, batch_size=128, epochs=100, file_name="sfullmodel.h5"):
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
        for i in range(len(pred[0])):
            if pred[0][i] == max(pred[0]): return i

    def graphing(self):
        loss = self.trained.history['loss']
        val_loss = self.trained.history['val_loss']
        epochs = np.arange(200)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        ds = Dataset("/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/UTKFace")
        ds.load(2)
        gm = Ethni_Model(ds)
        gm.load_model(file_name = '/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/sfullmodel.h5')
        gm.train(file_name="/Users/YuYinfeng/Desktop/Semester 4/CSE_204_ML/Project/sfullmodel.h5")

