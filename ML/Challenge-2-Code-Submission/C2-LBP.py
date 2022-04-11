# This is script which iterates through all the images and calls the necessary scripts to extract features and create the csv file 


import numpy as np 
#import pandas as pd 

import cv2
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

from skimage.feature import hog
from skimage.feature import local_binary_pattern
P = 8
R = 1 # check with the documentation, play with 3 and 1, and the p = n * r
bins = 10


def getAllRegions(image, img_s, filename, target_val):
    
    gray_seg = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
    # apply mask to original image
    mask_inv =  255 - mask;
    #result_img = mask_inv * gray_org
    #c = result_img
    
    train_set = np.array([])
    imageID = filename[-11:-4]
    temp = np.array([])
    temp = np.append(temp, imageID)
    
    result_img = mask_inv * image[:,:,0]
    c0 = result_img
    lbp       = local_binary_pattern(c0, P=P, R=R, method="uniform")
    lbp1, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
    
    
    result_img = mask_inv * image[:,:,1]
    c1 = result_img
    lbp       = local_binary_pattern(c1, P=P, R=R, method="uniform")
    lbp2, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
    
    
    result_img = mask_inv * image[:,:,2]
    c2 = result_img
    lbp       = local_binary_pattern(c2, P=P, R=R, method="uniform")
    lbp3, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
    
    
    lbp_all  = np.concatenate((lbp1, lbp2, lbp3),axis=0)
    #print(lbp_c.shape)
    
    for i in range(0, lbp_all.shape[0], 1):
            temp=np.append(temp,lbp_all[i])
    
    temp=np.append(temp,target_val)
    
    return temp

######################################################
################# READING ALL IMAGES #################
######################################################
# val - bcc,bkl,mel
# train - mel, bkl, bcc

# bcc train -- done
# bkl train -- done
# mel train 
# bcc val 
# bkl val 
# mel val
    
# (0=bcc, 1=bkl, and 2=mel)
tv = "val"
l_type = "mel"
    
if(l_type == "bcc"):
    l_par = '0'
elif(l_type == "bkl"):
    l_par = '1'
elif(l_type == "mel"):
    l_par = '2'
    

    
with open('{0}_features-LBP-C2-{1}.csv'.format(l_type,tv), 'a+') as csvfile:
        
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
            
        
   
    
                   

    
    
    


