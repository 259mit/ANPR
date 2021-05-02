#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pandas as pd


# In[1]:


im = cv2.imread('/Users/home/Downloads/WhatsApp Image 2021-04-08 at 13.41.08.jpeg')
bbox, label, conf = cv.detect_common_objects(im)
output_image = draw_bbox(im, bbox, label, conf)
plt.figure(figsize = (50,20))
plt.imshow(output_image)
print(label)


# In[2]:


import numpy as np


# In[86]:


ROIs = np.array([[324, 345,  71,  39],
       [395, 428,  97,  44],
       [407, 481, 113,  58],
       [407, 582, 145,  60],
       [427, 536, 132,  50],
       [351, 645, 145,  62],
       [264, 672, 171,  83],
       [231, 699, 136,  92],
       [226, 352,  89,  33],
       [263, 440,  95,  38],
       [254, 569, 109,  51],
       [202, 601, 137,  55],
       [150, 625, 121,  61]])


# In[92]:


ROIs = np.array([[  63,  528,  114 ,  64],
 [ 161,  536 , 128 ,  51],
 [ 239 , 530  , 87 ,  41],
 [ 305  ,510 , 105 ,  57],
 [ 375,  504 , 126 ,  54],
 [ 488,  497 , 110 ,  54],
 [ 593,  487 , 119 ,  59],
 [ 717,  485 ,  96  , 51],
 [ 825,  480 , 100  , 54],
 [ 906,  473 ,  89 ,  45],
 [1001,  468 ,  80 ,  40],
 [ 114,  591 , 116 ,  63],
 [ 211,  585 , 119 ,  61],
 [ 300,  563 , 121 ,  64],
 [ 399,  560 , 101 ,  56],
 [ 484,  548 , 119 ,  68],
 [ 579 , 554 ,  95 ,  52],
 [ 652 , 550 , 110 ,  54],
 [ 741 , 532  ,101 ,  66],
 [ 822 , 515 , 121 ,  69],
 [ 925 , 528 , 110 ,  49],
 [ 978 , 524 , 120 ,  56],
 [1064,  495 , 129 ,  66],
 [1162,  478 , 101 ,  61],
 [1057 , 449 ,  87 ,  47]])


# In[93]:


im = cv2.imread('/Users/home/Downloads/WhatsApp Image 2021-04-08 at 17.45.46.jpeg')


# In[94]:


i = 1
label_list = []
for rect in ROIs:
    x1=rect[0]
    y1=rect[1]
    x2=rect[2]
    y2=rect[3]
    #crop roi from original image
    img_crop=im[y1:y1+y2,x1:x1+x2]
    imm = im
    cv2.rectangle(imm, (x1, y1), (x1+x2, y1+y2), (255,0,0), 2)
    #show cropped image
    #cv2.imshow("crop"+str(crop_number),img_crop)
    bbox, label, conf = cv.detect_common_objects(img_crop)
    label_list.append(label)
    plt.figure()
    plt.imshow(img_crop)
    plt.show()
    #output_image = draw_bbox(img_crop, bbox, label, conf)
    #cv2.imshow('Slot',output_image)
    print('Slot: ',i, ': ', rect,' ',label)
    i = i+1
    #save cropped image
    #cv2.imwrite("crop"+str(crop_number)+".jpeg",img_crop)
label_list_df = pd.DataFrame(label_list) 
vacant = list(label_list_df.loc[pd.isna(label_list_df[0]), :].index)
full = list(label_list_df.loc[pd.isna(label_list_df[0]) == False, :].index)


# In[95]:


from PIL import Image
plt.figure(figsize = (50,20))
plt.imshow(imm)
imm = Image.fromarray(imm)
imm.save('/Users/home/Downloads/BTech DS/BigDataProject/imm2.jpg')


# In[96]:


vacant


# In[ ]:




