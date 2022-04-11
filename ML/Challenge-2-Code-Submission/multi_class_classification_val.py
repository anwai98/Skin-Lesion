from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

# Define Important Library
#====================================================================
import pandas as pd
import numpy as np
import os
import sys
import csv
import pickle
import datatable as dt
import time


# Start from 1 always, no random state
#====================================================================
np.random.seed(1)

# library import
#====================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import make_pipeline

import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

# Classifiers
#====================================================================
Classifiers = [
               
               #svm.LinearSVC(), #*
               #LogisticRegression(C=1, multi_class='multinomial'), #*
               #RidgeClassifier(), #*
               svm.SVC(kernel='rbf', C=10, probability=True, gamma = 0.0030), # too slow #*
               #RandomForestClassifier(random_state=0, criterion='gini'),
               
               #QuadraticDiscriminantAnalysis(),
               #BernoulliNB(),
               #DecisionTreeClassifier(random_state=0),
               #ExtraTreesClassifier(random_state=0, criterion='gini'),
               #GaussianNB(),
               #KNeighborsClassifier(n_neighbors=3),
               #LinearDiscriminantAnalysis(),
               #GradientBoostingClassifier(random_state=0), # too slow
               #AdaBoostClassifier(random_state=0)
               
               ]

# Validation Dataset 
#====================================================================
for classifier in Classifiers:
    
    print("Validation Dataset")
    print("-------------------------------------------------------")

    dt_df = dt.fread('csv_files/bcc_features-GLCM-C2-val.csv')
    df1 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/bkl_features-GLCM-C2-val.csv')
    df2 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/mel_features-GLCM-C2-val.csv')
    df3 = dt_df.to_pandas()
    
    imgID1 = np.asarray(df1.iloc[:, 0])
    imgID2 = np.asarray(df2.iloc[:, 0])
    imgID3 = np.asarray(df3.iloc[:, 0])
    imageID = np.concatenate((imgID1, imgID2, imgID3), axis=0)
    #print(imageID)
    
    train1_data = np.asarray(df1.iloc[:, 1:-1])
    train1_target = np.asarray(df1.iloc[:,-1]).reshape(-1, 1)

    train2_data = np.asarray(df2.iloc[:, 1:-1])
    train2_target = np.asarray(df2.iloc[:,-1]).reshape(-1, 1)

    train3_data = np.asarray(df3.iloc[:, 1:-1])
    train3_target = np.asarray(df3.iloc[:,-1]).reshape(-1, 1)

    # dt_df = dt.fread('csv_files/bcc_features-HOG-C2-val.csv')
    # df4 = dt_df.to_pandas()
    # dt_df = dt.fread('csv_files/bkl_features-HOG-C2-val.csv')
    # df5 = dt_df.to_pandas()
    # dt_df = dt.fread('csv_files/mel_features-HOG-C2-val.csv')
    # df6 = dt_df.to_pandas()
    # print(df4.shape)
    # print(df5.shape)
    # print(df6.shape)

    dt_df = dt.fread('csv_files/bcc_features-LBP-C2-val.csv')
    df7 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/bkl_features-LBP-C2-val.csv')
    df8 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/mel_features-LBP-C2-val.csv')
    df9 = dt_df.to_pandas()

    # dt_df = dt.fread('csv_files/bcc_features-Gabor-C2-val.csv')
    # df10 = dt_df.to_pandas()
    # dt_df = dt.fread('csv_files/bkl_features-Gabor-C2-val.csv')
    # df11 = dt_df.to_pandas()
    # dt_df = dt.fread('csv_files/mel_features-Gabor-C2-val.csv')
    # df12 = dt_df.to_pandas()
    # print(df10.shape)
    # print(df11.shape)
    # print(df12.shape)

    dt_df = dt.fread('csv_files/bcc_features-color-C2-val.csv')
    df13 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/bkl_features-color-C2-val.csv')
    df14 = dt_df.to_pandas()
    dt_df = dt.fread('csv_files/mel_features-color-C2-val.csv')
    df15 = dt_df.to_pandas()

    # train4_data = np.asarray(df4.iloc[:, 1:-1])
    # train5_data = np.asarray(df5.iloc[:, 1:-1])
    # train6_data = np.asarray(df6.iloc[:, 1:-1])

    train7_data = np.asarray(df7.iloc[:, 1:-1])
    train8_data = np.asarray(df8.iloc[:, 1:-1])
    train9_data = np.asarray(df9.iloc[:, 1:-1])

    # train10_data = np.asarray(df10.iloc[:, 1:-1])
    # train11_data = np.asarray(df11.iloc[:, 1:-1])
    # train12_data = np.asarray(df12.iloc[:, 1:-1])

    train13_data = np.asarray(df13.iloc[:, 1:-1])
    train14_data = np.asarray(df14.iloc[:, 1:-1])
    train15_data = np.asarray(df15.iloc[:, 1:-1])

    train1_data = np.concatenate((train1_data, train7_data, train13_data), axis=1)
    train2_data = np.concatenate((train2_data, train8_data, train14_data), axis=1)
    train3_data = np.concatenate((train3_data, train9_data, train15_data), axis=1)

    train4_data = np.vstack((train1_data,train2_data))
    train4_target = np.vstack((train1_target,train2_target))

    X_val = np.vstack((train3_data,train4_data))
    y_val = np.vstack((train3_target,train4_target))

    print(X_val.shape)
    print(y_val.shape)
    #print(np.unique(y))
    
    start = time.time()

    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)

    filename = classifier.__class__.__name__ + ".sav"
    [scaler, model, sel] = pickle.load(open(filename, 'rb'))
    
    # Scaling the data
    #====================================================================
    X_val = scaler.transform(X_val)
    X_val = sel.transform(X_val)

    #sampling = SMOTE('auto')
    #XX_val, yy_val = sampling.fit_resample(X_val, y_val)
    
    print("Validation Set Performance:")

    y_pred_val = model.predict(X_val)
    
    acc_val = accuracy_score(y_pred_val, y_val)
    kappa_val = cohen_kappa_score(y_pred_val, y_val)

    print('Acc: {0:3.4}%'.format(acc_val*100))
    print('kappa: {0:3.4}'.format(kappa_val))
    
    end = time.time()
    print("Time: {0}".format(end-start))
    
    target_names = ['class 0', 'class 1', 'class 2']
    print(classification_report(y_val, y_pred_val, target_names=target_names))
    print(confusion_matrix(y_val, y_pred_val))
    
    print("\n\n")
    """
    with open('test_val.csv', 'a+') as csvfile:
        for i in range(0,len(y_pred_val),1):
            csvfile.write("{0}\n".format(y_pred_val[i]))
    """
