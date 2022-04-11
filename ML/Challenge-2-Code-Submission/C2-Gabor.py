# This is script which iterates through all the images and calls the necessary scripts to extract features and create the csv file 


import numpy as np 
#import pandas as pd 

import cv2
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#from PIL import Image

import os
import sys
import csv
import string
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)

from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import integral_image
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage import io, color, img_as_ubyte


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
    imageID = filename[-11:-4]
    temp = np.array([])
    temp = np.append(temp, imageID)
    
    
    #Generate Gabor features
    
    ksize = 3  
    sigma = 1 #Large sigma on small features will fully miss the features. 
    lamda = 1*np.pi/4  
    gamma = 1 #Value of 1, spherical may not be ideal as it picks up features from other regions.
    phi = 0  #Phase offset
    
    for divide in range(1,4,1): 
         
        theta = 1 * np.pi/divide
        resizeK1 = cv2.resize(c, (200, 90)) 
        kernel1 = cv2.getGaborKernel((ksize,ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)  
        fimg_K1 = cv2.filter2D(resizeK1, cv2.CV_8UC3, kernel1) 
        #print(fimg_K1.shape)  
        filtered_img_K1 = fimg_K1.reshape(-1)
        #print(filtered_img_K1.shape)
        
        for i in range(0, filtered_img_K1.shape[0], 1):
                temp=np.append(temp,filtered_img_K1[i])
        
        #print(divide)
        #print(temp.shape)

    #print("line 146 {0}".format(temp.shape))
           
    temp=np.append(temp,target_val)
    
    return temp
    
   
######################################################
################# READING ALL IMAGES #################
######################################################
# val - bcc,bkl,mel
# train - mel, bkl, bcc

# bcc train -- 
# bkl train -- 
# mel train -- 
# bcc val 
# bkl val 
# mel val
    
# (0=bcc, 1=bkl, and 2=mel)
l_type = "bcc"
    
if(l_type == "bcc"):
    l_par = '0'
elif(l_type == "bkl"):
    l_par = '1'
elif(l_type == "mel"):
    l_par = '2'
    
tv = "val"
    
with open('{0}_features-Gabor-C2-{1}.csv'.format(l_type,tv), 'a+') as csvfile:
        
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
            
        
   
    
                   

    
    
    


