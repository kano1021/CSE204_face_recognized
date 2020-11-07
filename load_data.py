from PIL import Image
from random import randint
import os
import sys
import numpy as np
import cv2
 
IMAGE_SIZE = 128

def resize(image, height = IMAGE_SIZE, width = IMAGE_SIZE): # resize the image (to smaller image) and return it
    return cv2.resize(image, (height, width), interpolation=cv2.INTER_CUBIC)
 
def resize_capture(image, height = IMAGE_SIZE, width = IMAGE_SIZE): # resize the captured image
    top, bottom, left, right = 0, 0, 0, 0

    BLACK = [0, 0, 0]

    h, w, _ = image.shape  
    ma = max(h, w) # in case h and w are different, we make it a square
    mi = min(h, w)
    diff_hw = ma - mi
    if h == ma:
        if diff_hw%2 == 0:
            left += diff_hw//2
            right += diff_hw//2
        else:
            left += diff_hw//2
            right += diff_hw//2 + 1
    else:
        if diff_hw%2 == 0:
            top += diff_hw//2
            bottom += diff_hw//2
        else:
            top += diff_hw//2
            bottom += diff_hw//2 + 1
    temp = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    top, bottom, left, right = 0, 0, 0, 0

    if ma < IMAGE_SIZE:
        diff = IMAGE_SIZE - h
        if diff%2 == 0:
            top += diff//2
            bottom += diff//2
            left += diff//2
            right += diff//2
        else:
            top += diff//2
            bottom += diff//2 + 1
            left += diff//2
            right += diff//2 + 1
        
    if h == IMAGE_SIZE:
        pass
    if h > IMAGE_SIZE:
        temp = resize(temp)
    
    # add black edge to small pictures to enlarge it
    temp2 = cv2.copyMakeBorder(temp, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    # resize and return
    img_name = '%s/%d.jpg'%('images', randint(0,1000))                
    cv2.imwrite(img_name, temp2)

    return temp2

# load training data
images = []
a_labels =[] # age labels
g_labels = [] # gender labels
e_labels = [] # ethnicity labels

def load_from(path_name):    
    for item in os.listdir(path_name):
        # construct the full path
        full_path = os.path.abspath(os.path.join(path_name, item))
        
        if item.endswith('.jpg'):
            image = cv2.imread(full_path)                
            image = resize(image, IMAGE_SIZE, IMAGE_SIZE)
            
            images.append(image)
            counter = 0
            start = 0
            label = []
            a_label = [0,0,0,0,0,0,0,0,0]
            g_label = [0,0]
            e_label = [0,0,0,0,0]
            for i in range(len(item)):
                if item[i] == '_': 
                    counter += 1
                    label.append(int(item[start:i]))
                    start = i + 1
                if counter >= 3: break 
            age = label[0]
            if age in range(0,6):
                a_label[0] = 1
            if age in range(6,11):
                a_label[1] = 1
            if age in range(11,16):
                a_label[2] = 1
            if age in range(16,21):
                a_label[3] = 1
            if age in range(21,31):
                a_label[4] = 1
            if age in range(31,41):
                a_label[5] = 1
            if age in range(41,61):
                a_label[6] = 1
            if age in range(61,81):
                a_label[7] = 1
            if age > 80:
                a_label[8] = 1
            g_label[label[1]] = 1
            e_label[label[2]] = 1
            a_labels.append(a_label)
            g_labels.append(g_label)
            e_labels.append(e_label)

    return np.array(images), np.array(a_labels), np.array(g_labels), np.array(e_labels)
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, a_labels, g_labels, e_labels = load_from("UTKFace")
        print(g_labels[:20])
