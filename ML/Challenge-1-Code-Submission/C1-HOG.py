# This is script which iterates through all the images and calls the necessary scripts to extract features and create the csv file 


import numpy as np 
#import pandas as pd 

import cv2
from skimage.feature import hog
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
"""
#from PIL import Image

import os
import sys
import csv
import string
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)


def getAllRegions(image, img_s, filename, target_val):
    
    # convert to grayscale
    print(image.shape)
    gray_org = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_seg = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
    # apply mask to original image
    mask_inv =  255 - mask;
    result_img = mask_inv * gray_org
    c = result_img
    #result_img = cv2.bitwise_and(mask, gray_org)
    #print(np.unique(result_img))

    train_set = np.array([])
    imageID = filename[-10:-4]
    temp = np.array([])
    temp = np.append(temp, imageID)
    
    #Generate HOG features
    fd, hog_image = hog(c, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=False)
    #print(fd.shape)
    #print(hog_image.shape)
    
    for i in range(0, fd.shape[0], 1):
            temp=np.append(temp,fd[i])
    
    """plt.figure
    plt.imshow(hog_image)
    plt.axis("off")
    plt.show()
    """
    
    temp=np.append(temp,target_val)
    
    return temp

######################################################
################# READING ALL IMAGES #################
######################################################

# train ls -- done
# train nv
# val ls
# val nv 

tv = "val"    
l_type = "nv"
    
if(l_type == "ls"):
    l_par = '1'
elif(l_type == "nv"):
    l_par = '0'
    
with open('{0}_features-HOG-C1-{1}.csv'.format(l_type,tv), 'a+') as csvfile:
        
    for i in range(0,7000,1):
        
        if ((i>=1) and (i<=9)):
            s = "000" + str(i)
        elif ((i>=10) and (i<=99)):
            s = "00" + str(i)
        elif ((i>=100) and (i<=999)):
            s = "0" + str(i)
        else:
            s = str(i)
            
        filename_org = l_type + s + ".jpg"
        filename_s = "s_" + l_type + s + ".jpg"
        
        print("----- Working on {0}....... --------".format(filename_org))
        #print(filename_s)
    
        img = cv2.imread(os.path.join(dirname, '{0}'.format(tv), l_type , filename_org))
        img_s = cv2.imread(os.path.join(dirname, '{0}'.format(tv), '{0}_seg'.format(l_type), filename_s))
        
        if(type(img_s) != type(None)):
            df = getAllRegions(img, img_s, filename_org, l_par)
            print(df.shape)
            for i in range(0,df.shape[0],1):
                if(i!=0):
                    csvfile.write(",")
                csvfile.write("{0}".format(df[i]))
            csvfile.write("\n")
        
        
    
    
                   

    
    
    


