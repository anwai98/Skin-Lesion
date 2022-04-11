from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
#import pandas as pd
import datatable as dt
#import modin.pandas as pd
import numpy as np
import os
import sys
import csv
import pickle


# Start from 1 always, no random state
#====================================================================
np.random.seed(1)

# library import
#====================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import KFold


import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

# Define classifiers within a list
#====================================================================
Classifiers = [
               
               svm.SVC(kernel='rbf', C=10, probability=True), # **
               #RandomForestClassifier(random_state=0, criterion='gini'), # *
               #LogisticRegression(C=1) # **
               
               #RidgeClassifier(), # *
               #ExtraTreesClassifier(random_state=0, criterion='entropy'), # *
               #GradientBoostingClassifier(random_state=0), # *
               #KNeighborsClassifier(n_neighbors=7), #*
               #DecisionTreeClassifier(random_state=0),
               #GaussianNB(),
               #BernoulliNB(),
               #AdaBoostClassifier(random_state=0)
               ]
               

# Validation Dataset 
#====================================================================

dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)


for classifier in Classifiers:
    
    print("TEST SET")
    print("-------------------------------------------------------")

    dt_df = dt.fread('csv_files/xx_features-GLCM-C1.csv')
    df1 = dt_df.to_pandas()

    dt_df = dt.fread('csv_files/xx_features-LBP-C1.csv')
    df2 = dt_df.to_pandas()

    dt_df = dt.fread('csv_files/xx_features-color-C1.csv')
    df3 = dt_df.to_pandas()

    imageID = np.asarray(df1.iloc[:, 0])
    #print(imageID.shape)
    
    train1_data = np.asarray(df1.iloc[:, 1:])
    train2_data = np.asarray(df2.iloc[:, 1:])
    train3_data = np.asarray(df3.iloc[:, 1:])

    train_data = np.concatenate((train1_data, train2_data, train3_data), axis=1)
    X_val = train_data 

    print(X_val.shape)

    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)

    filename = classifier.__class__.__name__ + ".sav"
    [scaler, sel, model] = pickle.load(open(filename, 'rb'))
    
    # Scaling the data
    #====================================================================
    X_val = scaler.transform(X_val)
    X_val = sel.transform(X_val)

    y_pred_val = model.predict(X_val)
    
    with open('Challenge1-Predictions.csv', 'a+') as csvfile:
        for i in range(0,len(y_pred_val),1):
            csvfile.write("{0},{1}\n".format(imageID[i], y_pred_val[i]))
        
        
        
    
