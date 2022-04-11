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

from sklearn.model_selection import KFold


# TRAINING Dataset 
#====================================================================
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)

print("\nTRAINING Dataset")

dt_df = dt.fread('csv_files/bcc_features-GLCM-C2-train.csv')
df1 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/bkl_features-GLCM-C2-train.csv')
df2 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/mel_features-GLCM-C2-train.csv')
df3 = dt_df.to_pandas()
print(df1.shape)
print(df2.shape)
print(df3.shape)

train1_data = np.asarray(df1.iloc[:, 1:-1])
train1_target = np.asarray(df1.iloc[:,-1]).reshape(-1, 1)

train2_data = np.asarray(df2.iloc[:, 1:-1])
train2_target = np.asarray(df2.iloc[:,-1]).reshape(-1, 1)

train3_data = np.asarray(df3.iloc[:, 1:-1])
train3_target = np.asarray(df3.iloc[:,-1]).reshape(-1, 1)

# dt_df = dt.fread('csv_files/bcc_features-HOG-C2-train.csv')
# df4 = dt_df.to_pandas()
# dt_df = dt.fread('csv_files/bkl_features-HOG-C2-train.csv')
# df5 = dt_df.to_pandas()
# dt_df = dt.fread('csv_files/mel_features-HOG-C2-train.csv')
# df6 = dt_df.to_pandas()
# print(df4.shape)
# print(df5.shape)
# print(df6.shape)

dt_df = dt.fread('csv_files/bcc_features-LBP-C2-train.csv')
df7 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/bkl_features-LBP-C2-train.csv')
df8 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/mel_features-LBP-C2-train.csv')
df9 = dt_df.to_pandas()
print(df7.shape)
print(df8.shape)
print(df9.shape)

# dt_df = dt.fread('csv_files/bcc_features-Gabor-C2-train.csv')
# df10 = dt_df.to_pandas()
# dt_df = dt.fread('csv_files/bkl_features-Gabor-C2-train.csv')
# df11 = dt_df.to_pandas()
# dt_df = dt.fread('csv_files/mel_features-Gabor-C2-train.csv')
# df12 = dt_df.to_pandas()
# print(df10.shape)
# print(df11.shape)
# print(df12.shape)

dt_df = dt.fread('csv_files/bcc_features-color-C2-train.csv')
df13 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/bkl_features-color-C2-train.csv')
df14 = dt_df.to_pandas()
dt_df = dt.fread('csv_files/mel_features-color-C2-train.csv')
df15 = dt_df.to_pandas()
print(df13.shape)
print(df14.shape)
print(df15.shape)

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

X = np.vstack((train3_data,train4_data))
y = np.vstack((train3_target,train4_target))

print("----- The entire training dataset -----")
print(y.shape)
print(X.shape)
#print(np.unique(y))

print("\n\n")

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

# Spliting with 10-Folds :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)

# Scaling the data
#====================================================================
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# Perform Feature Selection
#====================================================================
#sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
# sel = SelectKBest(chi2, k='all') # keeps k features, need both X,y
sel = SelectFromModel(LinearSVC(C=0.5, penalty="l1", dual=False))
# sel = SelectFromModel(ExtraTreesClassifier())
#SelectFromModel(lsvc, prefit=True)
# sel = PCA(n_components=20)

X  = sel.fit_transform(X, y)
print("Performing Feature Selection ... ")
print(X.shape)


# Pick all classifier within the Classifier list and test one by one
#====================================================================

for classifier in Classifiers:

    start = time.time()
    
    y_all_proba = []
    y_all_test = []
    y_all_pred = []
    triple_neg_proba = []
    
    accuracy = []
    kappa = []
    auroc = []
    aupr = []
    
    fold = 1
    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)
    #model = OneVsOneClassifier(classifier)
    model = OneVsRestClassifier(classifier)
    
    for train_index, test_index in cv.split(X, y):
        
        X_train = X[train_index]
        X_test = X[test_index]
        
        y_train = y[train_index]
        y_test = y[test_index]
        
        # print the fold number and numbber of feature after selection
        # -----------------------------------------------------------------
        #print("----- Working on Fold -----------> {0}:".format(fold))
        
        # Applying Sampling to achieve class balance
        # -----------------------------------------------------------------
        # counting unique occurences in the target column
        (unique, counts) = np.unique(y_train, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        #print("Before")
        #print(frequencies)
        
        avg = 0
        for val in frequencies:
            avg = avg + val[1]
        avg = int(avg/3)
        #print(avg)
        
        # define sampling strategy
        strategy = dict()
        for val in frequencies:
            if(val[1] > avg):
                strategy[val[0]] = avg 
            else: 
                strategy[val[0]] = val[1]
        #print(strategy)
        
        # sampling on the chunk
        #sampling = RandomUnderSampler(sampling_strategy=strategy)
        sampling = SMOTE('auto')
        XX_train, yy_train = sampling.fit_resample(X_train, y_train)

        # Train model
        # -----------------------------------------------------------------
        model.fit(XX_train, yy_train)
        
        # Evaluation
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        #y_proba = model.predict_proba(X_test)
        #y_proba = np.nan_to_num(y_proba)

        #y_all_proba.append(y_proba)
        y_all_test.append(y_test)
        y_all_pred.append(y_pred)
        
        accuracy.append(accuracy_score(y_pred, y_test))
        kappa.append(cohen_kappa_score(y_test, y_pred))

        fold += 1
        
    end = time.time()
    
    filename = classifier.__class__.__name__ + ".sav"
    pickle.dump([scaler, model, sel], open(filename, 'wb'))
    
    y_all_test = np.concatenate(y_all_test)
    #y_all_proba = np.concatenate(y_all_proba)
    y_all_pred = np.concatenate(y_all_pred)

    #target_names = ['class 0', 'class 1', 'class 2']
    #print(classification_report(y_all_test, y_all_pred, target_names=target_names))
    
    print('Accuracy: {0:3.4}%'.format(np.mean(accuracy)*100))
    print('kappa: {0:3.4}'.format(np.mean(kappa)))
    print("Time: {0}".format(end-start))
    
    
    print('____________________________________________\n\n')



