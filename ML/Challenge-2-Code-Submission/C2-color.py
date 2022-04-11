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


from scipy.stats import skew, kurtosis

def statistical_features(input_image, mask_inv):
  
  # Mean Values of the Color Space
  mean_val1 = np.mean(mask_inv * input_image[:,:,0])
  mean_val2 = np.mean(mask_inv * input_image[:,:,1])
  mean_val3 = np.mean(mask_inv * input_image[:,:,2])
  # Standard Deviation of the Color Space
  std_val1 = np.std(mask_inv * input_image[:,:,0])
  std_val2 = np.std(mask_inv * input_image[:,:,1])
  std_val3 = np.std(mask_inv * input_image[:,:,2])
  # Variance
  var_val1 = np.var(mask_inv * input_image[:,:,0])
  var_val2 = np.var(mask_inv * input_image[:,:,1])
  var_val3 = np.var(mask_inv * input_image[:,:,2])
  # Skewness
  skew_val1 = skew((mask_inv * input_image[:,:,0]).reshape(-1))
  skew_val2 = skew((mask_inv * input_image[:,:,1]).reshape(-1))
  skew_val3 = skew((mask_inv * input_image[:,:,2]).reshape(-1))
  # Kurtosis
  kurt_val1 = kurtosis((mask_inv * input_image[:,:,0]).reshape(-1))
  kurt_val2 = kurtosis((mask_inv * input_image[:,:,1]).reshape(-1))
  kurt_val3 = kurtosis((mask_inv * input_image[:,:,2]).reshape(-1))

  def entropy_image(input_image):   
    hist = np.histogram(input_image, bins=256, range=(0,255), density=True)
    data = hist[0]/sum(hist[0])    
    ent = np.zeros((len(data)), dtype = float)
    for i in range(len(data)):
      if(data[i] == 0):
          ent[i] = 0;
      else:
          ent[i] = data[i]*np.log2(data[i]);     
    entropy = -(data*ent).sum()
    return entropy
    
  entropy_va11 = entropy_image(input_image[:,:,0])
  entropy_va12 = entropy_image(input_image[:,:,1])
  entropy_va13 = entropy_image(input_image[:,:,2])
  
  color_all = np.hstack([mean_val1, mean_val2, mean_val3, std_val1, std_val2, std_val3, var_val1, var_val2, var_val3, skew_val1, skew_val2, skew_val3, kurt_val1, kurt_val2, kurt_val3, entropy_va11, entropy_va12, entropy_va13])

  #18 values as output
  return color_all
  
  
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
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    stat_return = statistical_features(img_rgb, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    stat_return = statistical_features(img_lab, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    stat_return = statistical_features(img_ycrcb, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    stat_return = statistical_features(img_hsv, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
    img_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
    stat_return = statistical_features(img_luv, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
    img_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    stat_return = statistical_features(img_hls, mask_inv)
    temp = np.concatenate((temp,stat_return),axis=0)
    #print(temp.shape)
    
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
    

with open('{0}_features-color-C2-{1}.csv'.format(l_type,tv), 'a+') as csvfile:
        
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
            
        
   
    
                   

    
    
    


