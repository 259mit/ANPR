#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras_ocr
import matplotlib.pyplot as plt
import cv2


# In[2]:


import cv2 
import pytesseract
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from scipy import ndimage
from PIL import Image
import glob 
import os
import imutils
import easyocr
from tqdm import tqdm

import os
import fnmatch
import cv2
import numpy as np
import string
import time

import tensorflow as tf
from tensorflow.python.client import device_lib

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input,Add, Conv2D, MaxPool2D,UpSampling2D, Lambda, Bidirectional
from keras.models import Model
from tensorflow.keras import Model, Input, regularizers
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import pickle


# In[3]:


pipeline = keras_ocr.pipeline.Pipeline()

'/Users/home/Downloads/20.jpg',
                                          '/Users/home/Downloads/untitled folder 4/nump1.png',
    ,
                                          '/Users/home/Downloads/d2.jpg',
                                          '/Users/home/Downloads/electric_vehicles_india_ev_mandate_ev_green_number_plates_electric_car_rules_in_india_ev_rules_1556698450_1200x900.jpg',
                                          '/Users/home/Downloads/mcms.php.jpeg',
                                          '/Users/home/Downloads/skoda-india-front-license-plate.jpg'
# In[52]:


images = [
    keras_ocr.tools.read(img) for img in ['/Users/home/Downloads/20.jpg',
                                          '/Users/home/Downloads/untitled folder 4/nump1.png'
    ]
]


# In[53]:


images[0].shape


# In[54]:


prediction_groups = pipeline.recognize(images)


# In[24]:


len(prediction_groups[0])


# In[23]:


for i in range(1,len(prediction_groups[0])):
    print(prediction_groups[0][i][0])


# In[55]:


fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, 
                                    predictions=predictions, 
                                    ax=ax)


# In[65]:


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def DetectNumPlateNew(impath):
    img = cv2.imread(impath)
    #plt.imshow(img)
    img = cv2.resize(img, (500,400),interpolation = cv2.INTER_CUBIC )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)
    thresh = cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    screenCnt1 = screenCnt.reshape(4,2)
    gray2 = four_point_transform(gray, screenCnt1)
    #mask = np.zeros(gray.shape,np.uint8)
    #new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    #new_image = cv2.bitwise_and(img,img,mask=mask)
    # Now crop
    #(x, y) = np.where(mask == 255)
    #(topx, topy) = (np.min(x), np.min(y))
    #(bottomx, bottomy) = (np.max(x), np.max(y))
    #Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    ret, gray2 = cv2.threshold(gray2, 170, 255, cv2.THRESH_BINARY) 
    #Cropped = gray
    #plt.imshow(Cropped)
    #num_rows, num_cols = Cropped.shape[:2]
    #rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -10, 1)
    #img_rotation = cv2.warpAffine(Cropped, rotation_matrix, (num_cols, num_rows))
    #crop_img = Cropped[30:150, 40:355]
    #plt.imshow(Cropped)
    gray2 = cv2.resize(gray2, (160,56),interpolation = cv2.INTER_CUBIC )
    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = gray2
    #im = cv2.filter2D(im, -1, kernel)
    plt.imshow(gray2)
    #Cropped2 = np.expand_dims(Cropped2 , axis = 2)
    #text = pytesseract.image_to_string(im, config='--psm 11')
    #print("Detected license plate Number is:",text)
    return gray2


# In[66]:


im1 = '/Users/home/Downloads/20.jpg'
im2 = '/Users/home/Downloads/untitled folder 4/nump1.png'
im3 = '/Users/home/Downloads/d2.jpg'
im4 = '/Users/home/Downloads/b5438dd187fbe199f3ecaab6d7a47c0e.jpg'


# In[67]:


img1 = DetectNumPlateNew(im1)
gray = img1
color_img = cv2.merge([gray,gray,gray])
plt.imshow(color_img)


# In[68]:


color_img.shape


# In[69]:


img1 = [color_img, color_img]


# In[70]:


prediction_groups = pipeline.recognize(img1)


# In[71]:


prediction_groups


# In[51]:


for i in range(1,len(prediction_groups[0])):
    print(prediction_groups[0][i][0])


# In[58]:


a = 'hr26dq5551'
b = 'hr25d0551'
c = 'hr28d0551'

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# In[59]:


print(similar(a, b))
print(similar(a, c))


# In[ ]:




