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

from skimage.feature import greycomatrix, greycoprops
from skimage.transform import integral_image
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature
from skimage import io, color, img_as_ubyte


distances = [1, 2, 3]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
properties = ['contrast', 'energy', 'homogeneity', 'correlation']
theta = []


def getAllRegions(image, img_s, filename, target_val):
    
    # convert to grayscale
    gray_org = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_seg = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
    # apply mask to original image
    mask_inv =  255 - mask;
    result_img = mask_inv * gray_org
    c = result_img
    #result_img = cv2.bitwise_and(mask, gray_org)
    #print(np.unique(result_img))
    
    """
    plt.figure
    plt.imshow(result_img)
    plt.axis("off")
    plt.show()
    """
    
    train_set = np.array([])
    imageID = filename[-10:-4]
    temp = np.array([])
    temp = np.append(temp, imageID)
    
    #print("line 50 {0}".format(temp.shape))
    
    """c_area = cv2.contourArea(c)
    c_hull = cv2.convexHull(c)
    c_hull_area = cv2.contourArea(c_hull)
    if(c_area == 0):
        c_solidity = 0
    else:
        c_solidity = float(c_area)/c_hull_area
    """
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_org, mask)
    mean_val = cv2.mean(gray_org, mask)
    
    temp = np.append(temp, max_val)
    temp = np.append(temp, min_val)
    temp = np.append(temp, mean_val[0])
    
    
    """
    temp = np.append(temp, y)
    temp = np.append(temp, y+h)
    temp = np.append(temp, x)
    temp = np.append(temp, x+w)
    
    temp = np.append(temp, c_area)
    temp = np.append(temp, c_solidity)
    """
    
    glcm = greycomatrix(c, distances=distances, angles=angles,symmetric=True,normed=True)
            
    contrast = greycoprops(glcm, properties[0])
    energy = greycoprops(glcm, properties[1])
    homogeneity = greycoprops(glcm, properties[2])
    correlation = greycoprops(glcm, properties[3])
    
    
    for i in range(0, len(distances), 1):
        for j in range(0, len(angles), 1):
            #print(contrast[i][j])
            temp=np.append(temp,contrast[i][j])
            
    for i in range(0, len(distances), 1):
        for j in range(0, len(angles), 1):
            #print(contrast[i][j])
            temp=np.append(temp,energy[i][j])
            
    for i in range(0, len(distances), 1):
        for j in range(0, len(angles), 1):
            #print(contrast[i][j])
            temp=np.append(temp,homogeneity[i][j])
            
    for i in range(0, len(distances), 1):
        for j in range(0, len(angles), 1):
            #print(contrast[i][j])
            temp=np.append(temp,correlation[i][j])
    
    #print("line 146 {0}".format(temp.shape))
           
    temp=np.append(temp,target_val)
    
    return temp
    

    
######################################################
################# READING ALL IMAGES #################
######################################################

# train nv -- done
# train ls -- done
# val nv 
# val ls 

tv = "val"    
l_type = "nv"
    
if(l_type == "ls"):
    l_par = '1'
elif(l_type == "nv"):
    l_par = '0'
    
with open('{0}_features-GLCM-C1-{1}.csv'.format(l_type,tv), 'a+') as csvfile:
        
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
        
        
    
    
                   

    
    
    


