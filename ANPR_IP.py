#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[77]:


import os
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sb
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import imutils


# ## Loading and Processing the data

# In[2]:


num_char_data = pd.read_csv(r"C:\Users\Legion\Desktop\BDT Dataset\A_Z Handwritten Data.csv").astype('float32')


# In[3]:


num_char_data.shape


# In[4]:


num_char_data.rename(columns={'0':'label'}, inplace=True)


# In[5]:


num_char_data['label'].nunique()


# In[6]:


num_char_data


# In[7]:


X = num_char_data.drop('label',axis = 1)
y = num_char_data.label
#get the shape of labels and features 
print(f'Features SHAPE :{X.shape}')
print(f'Class Column SHAPE :{y.shape}')
#split into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[8]:


scaler = MinMaxScaler()
scaler.fit(X_train)
#scaling data 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[9]:


X_train = np.reshape(X_train, (X_train.shape[0], 28,28,1)).astype('float32')
X_test = np.reshape(X_test, (X_test.shape[0], 28,28,1)).astype('float32')
print("Train data shape: ", X_train.shape)
print("Test data shape: ", X_test.shape)


# In[10]:


y_train = np_utils.to_categorical(y_train,num_classes=36,dtype=int)
y_test = np_utils.to_categorical(y_test,num_classes=36,dtype=int)
y_train.shape,y_test.shape


# In[11]:


letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
             7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
             14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
             21:'V',22:'W',23:'X', 24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',
             34:'8',35:'9'}
#show 
fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train[i].reshape(28,28))
    ax.axis('off')
    ax.set(title = f"{letters_dict[y_train[i].argmax()]}")


# ## Model

# In[12]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
#output layer 
model.add(Dense(36,activation ="softmax"))
#compile 
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[13]:


model.summary()


# In[14]:


model.fit(X_train, y_train, epochs=5,batch_size=128,verbose=1,validation_data = (X_test,y_test))


# ## Evaluating the Model

# In[15]:


scores =model.evaluate(X_test,y_test,verbose=0)
print('Validation Loss : {:.2f}'.format(scores[0]))
print('Validation Accuracy: {:.2f}'.format(scores[1]))


# # Number Plate Recognition

# ## Making the necessary Functions

# In[67]:


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


# In[68]:


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


# In[69]:


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


# ## Recognizing the Number Plate

# In[71]:


img6, ogimg6 ,sp = DetectNumPlateNew(r"C:\Users\Legion\Downloads\WhatsApp Image 2021-04-09 at 9.02.28 PM.jpeg")
f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(ogimg6)
f.add_subplot(1,2, 2)
plt.imshow(img6)
plt.show(block=True)


# In[19]:


plt.imshow(img6)


# ## Character Recognition Of Number Plate

# In[20]:


gray = img6

#dilation
kernel = np.ones((1, 1), np.uint8)
img_dilation = cv2.dilate(gray, kernel, iterations=1)
#plt.imshow(img_dilation)
plt.imshow(gray)
#img_dilation = img1

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


# In[79]:


img69=a[2]
plt.imshow(img69)


# In[80]:


img70 = cv2.resize(img69 , (28 , 28))
img70=img70.reshape(1, 28,28,1)


# In[81]:


char_predict=model.predict(img70)
char_predict=list(char_predict)


# In[82]:


char_predict


# ### The index of the array with value as 1( it is binary) refers to number 2 hence correctly predicting our character

# In[ ]:


import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pandas as pd
import numpy as np


# In[ ]:


im = cv2.imread('/Users/home/Downloads/WhatsApp Image 2021-04-08 at 13.41.08.jpeg')


# In[ ]:


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


# In[ ]:


i = 1
label_list = []
for rect in ROIs:
    x1=rect[0]
    y1=rect[1]
    x2=rect[2]
    y2=rect[3]
    #crop roi from original image
    img_crop=im[y1:y1+y2,x1:x1+x2]
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

