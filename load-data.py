import scipy.io
from scipy import misc
import glob
import h5py
from os import rename
import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plt
from scipy.misc import imresize
import scipy.misc
import time
import random
from math import fsum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from time import perf_counter as timer
import pandas as pd
from skimage.util.shape import view_as_windows
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import median_filter
from scipy import ndimage
from skimage.filters import gaussian

#rename the files to be 00001.png, 00002.png, ... etc, instead of 1.png, 2.png, ..
#for fname in glob.glob("./train/*.png"):
#    if len(fname) != 17:
#        string1 = fname[8:]
#        string2 = fname[:8]
#        string3 = string1.zfill(9)
#        string4 = string2 + string3
#        rename(fname, string4)
#for fname in glob.glob("./test/*.png"):
#    if len(fname) != 16:
#        string1 = fname[7:]
#        string2 = fname[:7]
#        string3 = string1.zfill(9)
#        string4 = string2 + string3
#        rename(fname, string4)

#load the data

#the basic 32x32 images
train_house_number = scipy.io.loadmat('train_32x32.mat')
test_house_number = scipy.io.loadmat('test_32x32.mat')

#look at the data
print(train_house_number['X'].shape)
print(train_house_number['y'].shape)
print(test_house_number['X'].shape)
print(test_house_number['y'].shape)

#the training images, resized to 50x150
train_house_sequence = []
for image_path in sorted(glob.glob("./train/*.png")):
    image = misc.imread(image_path)
    image = misc.imresize(image, (50, 150, 3))
    train_house_sequence.append(image)
train_house_sequence = np.array(train_house_sequence)
print(train_house_sequence.shape)

#the testimg images, resized to 50x150
test_house_sequence = []
for image_path in sorted(glob.glob("./test/*.png")):
    image = misc.imread(image_path)
    image = misc.imresize(image, (50,150,3))
    test_house_sequence.append(image)
test_house_sequence = np.array(test_house_sequence)
print(test_house_sequence.shape)

#idea:
#rather than resizing the image to fit in a 50x150, should instead pad with zeros along the right side and the bottom.
#first step: look through the location info and see if there exists anything beyond 50 in the up/down direction and
#150 in the rightleft direction

testArray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(7 in testArray)

#if True, that means you can search through:
#test_house_sequence_y_width
test_house_sequence_y_left = np.array(test_house_sequence_y_left)
test_house_sequence_y_top = np.array(test_house_sequence_y_top)
train_house_sequence_y_left = np.array(train_house_sequence_y_left)
train_house_sequence_y_top = np.array(train_house_sequence_y_left)
test_house_sequence_y_width = np.array(test_house_sequence_y_width)
test_house_sequence_y_height = np.array(test_house_sequence_y_height)
train_house_sequence_y_width = np.array(train_house_sequence_y_width)
train_house_sequence_y_height = np.array(train_house_sequence_y_height)
#print(np.where(test_house_sequence_y_width))
#test_house_sequence_y_height
#train_house_sequence_y_width
#train_house_sequence_y_height
print(test_house_sequence_y_left.shape)
print(test_house_sequence_y_left[1])
print(test_house_sequence_y_width[1])


widthList = []
for i in range(len(test_house_sequence_y_left)):
    widthList.append([sum(x) for x in zip(test_house_sequence_y_left[i], test_house_sequence_y_width[i])])
    
heightList = []
for i in range(len(test_house_sequence_y_top)):
    heightList.append([sum(x) for x in zip(test_house_sequence_y_top[i], test_house_sequence_y_height[i])])

widthList = np.array(widthList)
heightList = np.array(heightList)

maxList = []
for i in range(len(widthList)):
    maxList.append(np.amax(np.array(widthList[i])))

print(max(maxList))

maxList2 = []
for i in range(len(heightList)):
    maxList2.append(np.amax(np.array(heightList[i])))
    
print(max(maxList2))

trainWidthList

#The Code Below is courtesy of https://discussions.udacity.com/t/how-to-deal-with-mat-files/160657/4
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label']  = pictDat[i]['label'][j]
                figure['left']   = pictDat[i]['left'][j]
                figure['top']    = pictDat[i]['top'][j]
                figure['width']  = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
        
dsf = DigitStructFile('./train/digitStruct.mat')
train_house_sequence_y = dsf.getAllDigitStructure_ByDigit() 

dsf2 = DigitStructFile('./test/digitStruct.mat')


test_house_sequence_y = dsf2.getAllDigitStructure_ByDigit() 

#my own code again
#save the label information
train_house_sequence_y_num = []
train_house_sequence_y_name = []
train_house_sequence_y_label = []
train_house_sequence_y_left = []
train_house_sequence_y_top = []
train_house_sequence_y_width = []
train_house_sequence_y_height = []

for i in range(len(train_house_sequence_y)):
    #store the number of digits
    n_digit = len(train_house_sequence_y[i]['boxes'])
    train_house_sequence_y_num.append(n_digit)
    #store the name of the file
    train_house_sequence_y_name.append(train_house_sequence_y[i]['filename'])
    
    label = []
    left = []
    top = []
    width = []
    height = []
    for j in range(n_digit):
        #store the label of the digits in a list
        label.append(train_house_sequence_y[i]['boxes'][j]['label'])
        #store the left side of the digit in a list
        left.append(train_house_sequence_y[i]['boxes'][j]['left'])
        #store the top of the digit in a list
        top.append(train_house_sequence_y[i]['boxes'][j]['top'])
        #store the width of the digit in a list
        width.append(train_house_sequence_y[i]['boxes'][j]['width'])
        #store the height of the digit in a list
        height.append(train_house_sequence_y[i]['boxes'][j]['height'])
        
    train_house_sequence_y_label.append(label)
    train_house_sequence_y_left.append(left)
    train_house_sequence_y_top.append(top)
    train_house_sequence_y_width.append(width)
    train_house_sequence_y_height.append(height)

    
test_house_sequence_y_num = []
test_house_sequence_y_name = []
test_house_sequence_y_label = []
test_house_sequence_y_left = []
test_house_sequence_y_top = []
test_house_sequence_y_width = []
test_house_sequence_y_height = []

for i in range(len(test_house_sequence_y)):
    #store the number of digits
    n_digit = len(test_house_sequence_y[i]['boxes'])
    test_house_sequence_y_num.append(n_digit)
    #store the name of the file
    test_house_sequence_y_name.append(test_house_sequence_y[i]['filename'])
    #store the label of the digits in a list
    label = []
    left = []
    top = []
    width = []
    height = []
    for j in range(n_digit):
        label.append(test_house_sequence_y[i]['boxes'][j]['label'])
        #store the left side of the digit in a list
        left.append(test_house_sequence_y[i]['boxes'][j]['left'])
        #store the top of the digit in a list
        top.append(test_house_sequence_y[i]['boxes'][j]['top'])
        #store the width of the digit in a list
        width.append(test_house_sequence_y[i]['boxes'][j]['width'])
        #store the height of the digit in a list
        height.append(test_house_sequence_y[i]['boxes'][j]['height'])
        
    test_house_sequence_y_label.append(label)
    test_house_sequence_y_left.append(left)
    test_house_sequence_y_top.append(top)
    test_house_sequence_y_width.append(width)
    test_house_sequence_y_height.append(height)
    
#make the 32x32 images/labels more usable
house_number_X_train = np.rollaxis(train_house_number['X'], 3)
print(house_number_X_train.shape)

house_number_y_train = train_house_number['y']
#print(house_number_y_train.shape)

house_number_X_test = np.rollaxis(test_house_number['X'], 3)
print(house_number_X_test.shape)

house_number_y_test = test_house_number['y']
#print(house_number_y_test.shape)

#one-hot encode the output labels
def onehot(array):
    """input is an array with a digit per row
    Output is an array, where each row is the same digit one-hot encoded"""
    new_array = []
    for i in range(len(array)):
        new_row = np.zeros(10)
        digit = array[i]
        if digit == 10:
            digit = 0
        new_row[digit] = 1
        new_array.append(new_row)
    return np.array(new_array)

house_number_y_train = onehot(house_number_y_train)
print(house_number_y_train.shape)
house_number_y_test = onehot(house_number_y_test)
print(house_number_y_test.shape)

train_house_sequence_y_num = np.array(train_house_sequence_y_num)
train_house_sequence_y_name = np.array(train_house_sequence_y_name)
train_house_sequence_y_label = np.array(train_house_sequence_y_label)
test_house_sequence_y_num = np.array(test_house_sequence_y_num)
test_house_sequence_y_name = np.array(test_house_sequence_y_name)
test_house_sequence_y_label = np.array(test_house_sequence_y_label)
