#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[22]:


import os
import fnmatch
import cv2
import numpy as np
import string
import time

import tensorflow as tf
from tensorflow.python.client import device_lib

from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, LSTM,GlobalAveragePooling2D, Reshape,Dropout, BatchNormalization,Activation, Input,Add, Conv2D,Flatten, MaxPool2D, MaxPooling2D,UpSampling2D, Lambda, Bidirectional
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


# In[4]:


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


# In[4]:


img = cv2.imread('/Users/home/Downloads/20.jpg')


# In[229]:


pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'


# In[5]:


def DetectNumPlate(impath):
    img = cv2.imread(impath)
    plt.imshow(img)
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
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]
    ret, thresh1 = cv2.threshold(Cropped, 120, 255, cv2.THRESH_BINARY) 
    #Cropped = gray
    #plt.imshow(Cropped)
    #num_rows, num_cols = Cropped.shape[:2]
    #rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -10, 1)
    #img_rotation = cv2.warpAffine(Cropped, rotation_matrix, (num_cols, num_rows))
    #crop_img = Cropped[30:150, 40:355]
    #plt.imshow(Cropped)
    Cropped2 = cv2.resize(thresh1, (80,28),interpolation = cv2.INTER_CUBIC )
    plt.imshow(Cropped2)
    Cropped2 = np.expand_dims(Cropped2 , axis = 2)
    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected license plate Number is:",text)
    return Cropped2, text


# In[128]:


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
    #edged = impath
    plt.imshow(img)
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
    #gray2 = cv2.GaussianBlur(gray2,(7,7),0)
    ret, gray2B = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY_INV) 
    #kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #gray2 = cv2.morphologyEx(gray2, cv2.MORPH_DILATE, kernel3)
    #Cropped = gray
    #plt.imshow(Cropped)
    #num_rows, num_cols = Cropped.shape[:2]
    #rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -10, 1)
    #img_rotation = cv2.warpAffine(Cropped, rotation_matrix, (num_cols, num_rows))
    #crop_img = Cropped[30:150, 40:355]
    #plt.imshow(Cropped)
    gray2 = cv2.resize(gray2B, (250,69),interpolation = cv2.INTER_CUBIC )
    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = gray2
    #im = cv2.filter2D(im, -1, kernel)
    #print(plt.imshow(gray2))
    #plt.imshow(img)
    #Cropped2 = np.expand_dims(Cropped2 , axis = 2)
    #text = pytesseract.image_to_string(im, config='--psm 11')
    #print("Detected license plate Number is:",text)
    return gray2, img, gray2B.shape #, gray2B


# In[191]:


im1 = '/Users/home/Downloads/20.jpg'
im2 = '/Users/home/Downloads/untitled folder 4/nump1.png'
im3 = '/Users/home/Downloads/d2.jpg'
im4 = '/Users/home/Downloads/b5438dd187fbe199f3ecaab6d7a47c0e.jpg'
#img4, og4, sp = DetectNumPlateNew(im4)
im5 = '/Users/home/Downloads/Indian_Vehicle_Registration_Plate_-_Kolkata_2011-07-29_4088.JPG'
im6 = '/Users/home/Downloads/Ra273c0317fede42978f21fc490abb5b7.jpeg'
im10 = '/Users/home/Downloads/images.jpeg'
im11 = '/Users/home/Downloads/2014-Hyundai-Elite-i20-Review-6-1024x683.jpg.webp'
img12 = '/Users/home/Downloads/Mercedes-Benz-CLS-India-Roadtest-54976.jpg'
im7 = '/Users/home/Downloads/WhatsApp Image 2021-03-29 at 23.16.48.jpeg'
im9 = '/Users/home/Downloads/main-qimg-d88a25f2330486be1a258918f4706bcd.jpeg'
im12 = '/Users/home/Downloads/vehicles-will-soon-come-fitted-with-number-plates6-1554372167.jpg'
im13 = '/Users/home/Downloads/chrome-car-number-plate-500x500-2.jpg'
im14 = '/Users/home/Downloads/d7e06425bcb8ae719796190434d1b649.jpg'
im15 = '/Users/home/Downloads/ezgif.com-webp-to-jpg-3.jpg.webp'
im16 = '/Users/home/Downloads/Car-Image-with-Indian-Number-Plate.png'
im17 = '/Users/home/Downloads/skoda-india-front-license-plate-2.jpg'
im18 = '/Users/home/Downloads/Vehicles.jpg'
im19 = '/Users/home/Downloads/mcms.php-2.jpeg'
im20 = '/Users/home/Downloads/main-qimg-b8771d5d1c5ca662241d59f8a5806488.jpeg'
im21 = '/Users/home/Downloads/34b0536dc5b84dcba250ab2565f225f7.jpg'
im22 = '/Users/home/Downloads/20-1461155101-blue-2.jpg'
im23 = '/Users/home/Downloads/9b138d6dc3d648a0b5778adb91b7cad2.JPG'
im24 = '/Users/home/Downloads/main-qimg-18d6ae6c83fbebdab97653a14b6ce46c.jpeg'
im25 = '/Users/home/Downloads/karun-car-main.jpg' 
im26 = '/Users/home/Downloads/weather_a751d4dc-2461-11e7-a4a0-8e0501b9fa54.jpg'
im27 = '/Users/home/Downloads/Porsche-718-Boxster-Registration.jpg'
im28 = '/Users/home/Downloads/computerized-printed-car-number-plate-500x500.jpg'
im29 = '/Users/home/Downloads/images-3.jpeg'
im30 ='/Users/home/Downloads/2018-Maruti-Swift-test-drive-review-front-three-quarters-view.jpg'
im31 = '/Users/home/Downloads/Screenshot 2021-04-07 at 8.32.15 PM.png'
im33 = '/Users/home/Downloads/Singh-S11-Nhc-Private-Number-Plate-Indian-Asain.jpg'
im34 = '/Users/home/Downloads/2018-Toyota-Yaris-Review-India-10.jpg'
im32 = '/Users/home/Downloads/indian-car-carriers-gurgaon-sector-17-gurgaon-car-hire-for-outstation-2vx90pv.jpg'
im8 = '/Users/home/Downloads/Samples-of-Characters-for-Recognition-Sample-Indian-Vehicle-Sample-Foreign-Vehicle_Q320.jpg'


# In[227]:


img6, ogimg6 ,sp = DetectNumPlateNew(im11)
f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(ogimg6)
f.add_subplot(1,2, 2)
plt.imshow(img6)
plt.show(block=True)
im6 = Image.fromarray(img6)
oim6 = Image.fromarray(ogimg6)
oim6.save("/Users/home/Downloads/BTech DS/BigDataProject/drive/cars/imc10.jpg")
im6.save("/Users/home/Downloads/BTech DS/BigDataProject/drive/cars/imp10.jpg")


# In[64]:





# In[337]:


plt.imshow(og1)


# In[10]:


plt.imshow(ogimg1)


# In[339]:


img1 = DetectNumPlate(im2)


# In[9]:


img1, ogimg1, sp = DetectNumPlateNew(im2)


# In[11]:


img1.shape


# In[15]:


img2 = DetectNumPlateNew(crop_image(img1,tol=254))


# In[381]:


img4, a, b = DetectNumPlateNew(im4)


# In[13]:


def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


# In[16]:


plt.imshow(crop_image(gray2,tol=10))


# In[349]:


img5, ogimg5, sp = DetectNumPlateNew(im5)


# In[1]:


'1'


# In[165]:


plt.imshow(ogimg5)


# In[30]:


img3


# In[66]:


gray = img6


# In[67]:


#dilation
kernel = np.ones((1, 1), np.uint8)
img_dilation = cv2.dilate(gray, kernel, iterations=1)
#plt.imshow(img_dilation)
plt.imshow(gray)
#img_dilation = img1


# In[68]:


a=[]
#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    if 850>cv2.contourArea(ctr)>299 :
        
        x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
        roi = gray[y:y+h, x:x+w]
    # show ROI
        a.append(roi)
        #cv2.rectangle(gray,(x,y),( x + w, y + h ),(90,0,255),2)
        plt.imshow(gray)
        #cv2.waitKey(0)


# In[70]:


plt.imshow(a[0])


# In[230]:


pytesseract.image_to_string(a[0], config='--psm 11')


# In[79]:


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = gray

# Convert to float type
pixel_vals = np.float32(pixel_vals)


image = gray
backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
plt.imshow(backtorgb)


# In[109]:


#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initally chosed for k-means clustering
k = 10
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))


# In[113]:


img = cv2.imread(im1)
plt.imshow(img)
_, thresholded = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
amount, labels = cv2.connectedComponents(thresholded)

# lets draw it for visualization purposes
preview = np.zeros((img.shape[0], img.shape[2], 3), dtype=np.uint8)


# In[80]:


from skimage import measure


# In[84]:


# perform a connected components analysis and initialize the mask to store the locations
		# of the character candidates
labels = measure.label(gray, neighbors=8, background=0)
charCandidates = np.zeros(gray.shape, dtype="uint8")
labels


# In[85]:


# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue

	# otherwise, construct the label mask to display only connected components for the
	# current label, then find contours in the label mask
	labelMask = np.zeros(gray.shape, dtype="uint8")
	labelMask[labels == label] = 255
	cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]


# In[87]:





# In[ ]:


plate_image = cv2.convertScaleAbs(Lp[0], alpha=(255.0))


# In[74]:


LpImg = img1
binary = img1b
plate_image = img1


# In[28]:


if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    #plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    #gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(LpImg,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ## Applied dilation 
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)


# In[29]:


plt.imshow(thre_mor)


# In[77]:


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
            
            # Sperate number and gibe prediction
            curr_num = binary[y:y+h,x:x+w] #thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)
            
print("Detect {} letters...".format(len(crop_characters))) 
plt.imshow(test_roi)


# In[76]:


fig = plt.figure(figsize=(14,4))
grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")


# In[ ]:


gray 
gray = Cropped2

#grayscale
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow('gray',gray)


# In[ ]:


#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)
cv2.waitKey(0)


# In[23]:


#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(img3, kernel, iterations=1)
plt.imshow(img_dilation)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    if 5000>cv2.contourArea(ctr)>100 :
        
        x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
        roi = gray[y:y+h, x:x+w]
    # show ROI
        a.append(roi)
        #cv2.rectangle(gray,(x,y),( x + w, y + h ),(90,0,255),2)
        #cv2.imshow('marked areas',gray)
        #cv2.waitKey(0)


# In[137]:


gray = img3
edged = cv2.Canny(gray, 30, 200)
thresh = cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
plt.imshow(thresh)


# In[138]:


contours=cv2.findContours(thresh.copy(),cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]


# In[139]:


contours[2].shape


# In[141]:


screenCnt = None
for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    #print(0.018 * peri)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    #print(approx)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
screenCnt1 = screenCnt.reshape(4,2)
gray2 = four_point_transform(gray, screenCnt1)


# In[158]:


plt.imshow(img5)


# In[159]:


binary = img5
plate_image = ogimg5
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 17, 37

for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)
            
print("Detect {} letters...".format(len(crop_characters)))


# In[527]:


imagehs = '/Users/home/Downloads/Screenshot 2021-03-30 at 6.22.08 PM.png'


# In[528]:


img = cv2.imread(imagehs)
mask = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1][:,:,0]
dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)


# In[530]:


crosses = mask[235:267,290:320] | mask[233:265,288:318]
mask[235:267,288:318] = crosses
dst = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)


# In[531]:


plt.imshow(dst)


# In[ ]:





# In[ ]:





# In[9]:


model = tf. keras.Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())


# In[10]:


model.summary()


# In[13]:


model = tf. keras.Sequential()
model.add(Dense(1024, input_shape=(3072, )))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[14]:


model.summary()


# In[28]:


model = tf. keras.Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),
                 activation ='relu', input_shape = (32,32,3)))


# In[29]:


model.summary()


# In[ ]:





# In[23]:


basemodel = InceptionResNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256,256,3)))
basemodel.trainable = False
headmodel = basemodel.output
headmodel = GlobalAveragePooling2D(name = 'global_average_pool')(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(256, activation = "relu", name = 'dense_1')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = "relu", name = 'dense_2')(headmodel)
headmodel = Dropout(0.2)(headmodel)
headmodel = Dense(11, activation = 'softmax', name = 'dense_3')(headmodel)
model = Model(inputs = basemodel.input, outputs = headmodel)
model.summary()


# In[24]:


basemodel = InceptionResNetV2(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256,256,3)))
basemodel.trainable = False
headmodel = basemodel.output
headmodel = GlobalAveragePooling2D(name = 'global_average_pool')(headmodel)
headmodel = Flatten(name= 'flatten')(headmodel)
headmodel = Dense(256, activation = "relu", name = 'dense_1')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = "relu", name = 'dense_2')(headmodel)
headmodel = Dropout(0.2)(headmodel)
headmodel = Dense(11, activation = 'softmax', name = 'dense_3')(headmodel)
model = Model(inputs = basemodel.input, outputs = headmodel)
model.summary()


# In[26]:


a = np.array([0.5666, 0.555, 0.333])
(a>0.5).astype("Int32")


# In[ ]:




