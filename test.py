import os
import cv2
import sys
from PIL import Image
from keras.models import load_model
import model
from load_data import IMAGE_SIZE, resize

g_model = model.Gender_Model(None)
g_model.load_model(file_name = 'gender_classifier.h5')
print('g_model loaded')

for item in os.listdir("crop_part1_super_dev"):
    #从初始路径开始叠加，合并成可识别的操作路径
    full_path = os.path.abspath(os.path.join("crop_part1_gender", item))
    if item.endswith('.jpg'):
        image = cv2.imread(full_path) 
        image = resize(image)
        gender = g_model.predict(image)
        if gender != 0: print(gender, item)
        