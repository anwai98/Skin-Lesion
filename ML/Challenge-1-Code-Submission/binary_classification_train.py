from __future__ import division

# By pass warnings
#====================================================================
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn

print("\n------ TRAINING-Classification.py ------")


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
import time


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


import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image


# TRAINING Dataset 
#====================================================================
dirname = os.path.dirname(__file__)
np.set_printoptions(threshold=sys.maxsize)

print("\n\n-------------------------------------------------------")
print("TRAINING Dataset")
print("-------------------------------------------------------")

dt_df = dt.fread('csv_files/ls_features-GLCM-C1-train.csv')
df1 = dt_df.to_pandas()
print(df1.shape)

dt_df = dt.fread('csv_files/nv_features-GLCM-C1-train.csv')
df2 = dt_df.to_pandas()
print(df2.shape)

"""dt_df = dt.fread('csv_files/ls_features-HOG-C1-train.csv')
df3 = dt_df.to_pandas()
print(df3.shape)

dt_df = dt.fread('csv_files/nv_features-HOG-C1-train.csv')
df4 = dt_df.to_pandas()
print(df4.shape)"""

dt_df = dt.fread('csv_files/ls_features-LBP-C1-train.csv')
df5 = dt_df.to_pandas()
print(df5.shape)

dt_df = dt.fread('csv_files/nv_features-LBP-C1-train.csv')
df6 = dt_df.to_pandas()
print(df6.shape)

"""dt_df = dt.fread('csv_files/ls_features-Gabor-C1-train.csv')
df7 = dt_df.to_pandas()
print(df7.shape)

dt_df = dt.fread('csv_files/nv_features-Gabor-C1-train.csv')
df8 = dt_df.to_pandas()
print(df8.shape)"""

dt_df = dt.fread('csv_files/ls_features-color-C1-train.csv')
df9 = dt_df.to_pandas()
print(df9.shape)

dt_df = dt.fread('csv_files/nv_features-color-C1-train.csv')
df10 = dt_df.to_pandas()
print(df10.shape)


train1_data = np.asarray(df1.iloc[:, 1:-1])
train1_target = np.asarray(df1.iloc[:,-1]).reshape(-1, 1)

train2_data = np.asarray(df2.iloc[:, 1:-1])
train2_target = np.asarray(df2.iloc[:,-1]).reshape(-1, 1)

print(train1_data.shape)
print(train2_data.shape)

#train3_data = np.asarray(df3.iloc[:, 1:-1])
#train4_data = np.asarray(df4.iloc[:, 1:-1])
train5_data = np.asarray(df5.iloc[:, 1:-1])
train6_data = np.asarray(df6.iloc[:, 1:-1])
#train7_data = np.asarray(df7.iloc[:, 1:-1])
#train8_data = np.asarray(df8.iloc[:, 1:-1])
train9_data = np.asarray(df9.iloc[:, 1:-1])
train10_data = np.asarray(df10.iloc[:, 1:-1])

train1_data = np.concatenate((train1_data, train5_data, train9_data), axis=1)
train2_data = np.concatenate((train2_data, train6_data, train10_data), axis=1)

X = np.vstack((train1_data,train2_data))
y = np.vstack((train1_target,train2_target))
y = y.astype(int)

print("----- The entire dataset -----")
print(X.shape)
print(y.shape)
#print(np.unique(y))

#(unique, counts) = np.unique(y, return_counts=True)
#print(unique)
#print(counts)

# Scaling the data
#====================================================================
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Features Selection
#====================================================================
#sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
#sel = SelectKBest(chi2, k=15) # keeps k features, need both X,y
sel = SelectFromModel(LinearSVC(C = 0.09, penalty="l1",  dual=False)) # need both X,y
# sel = SelectFromModel(ExtraTreesClassifier())
#sel = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3))
#sel = RFE(estimator=svc, n_features_to_select=1, step=1)
#SelectFromModel(lsvc, prefit=True)
#sel = PCA(n_components=500)

print("Performing Feature Selection ... ")
X = sel.fit_transform(X, y)
print(X.shape)

print("\n\n")

# Define classifiers within a list
#====================================================================
Classifiers = [
               
               svm.SVC(kernel='rbf', C=10, probability=True), # **
               #RandomForestClassifier(random_state=0, criterion='gini'), # *
               #LogisticRegression(C=1), # **
               
               #RidgeClassifier(), # *
               #ExtraTreesClassifier(random_state=0, criterion='entropy'), # *
               #GradientBoostingClassifier(random_state=0), # *
               #KNeighborsClassifier(n_neighbors=7), #*
               #DecisionTreeClassifier(random_state=0),
               #GaussianNB(),
               #BernoulliNB(),
               #AdaBoostClassifier(random_state=0),
               #LinearDiscriminantAnalysis(),
               #QuadraticDiscriminantAnalysis()
               ]

# Spliting with 10-Folds :
#====================================================================
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True)


# Pick all classifier within the Classifier list and test one by one
#====================================================================

print("In the classification :")

for classifier in Classifiers:
    
    start = time.time()

    y_all_test = []
    y_all_proba = []
    accuracy = []
    auroc = []
    aupr = []
    kappa = []
    fold = 1

    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)
    model = classifier

    for train_index, test_index in cv.split(X, y):

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Train model
        # -----------------------------------------------------------------
        model.fit(X_train, y_train)

        # Evalution
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        #y_proba = model.predict_proba(X_test)[:, 1]

        y_all_test.append(y_test)
        #y_all_proba.append(y_proba)

        accuracy.append(accuracy_score(y_pred=y_pred, y_true=y_test))
        kappa.append(cohen_kappa_score(y_test, y_pred))
        #auroc.append(roc_auc_score(y_true=y_test, y_score=y_proba))
        #aupr.append(average_precision_score(y_true=y_test, y_score=y_proba))

        fold += 1
    
    end = time.time()
    
    filename = classifier.__class__.__name__ + ".sav"
    pickle.dump([scaler, sel, model], open(filename, 'wb'))
    
    print("")
    print('Acc: {0:3.4}%'.format(np.mean(accuracy)*100))
    #print('auROC: {0:3.4}'.format(np.mean(auroc)))
    #print('auPR: {0:3.4}'.format(np.mean(aupr)))
    print('kappa: {0:3.4}'.format(np.mean(kappa)))
    print("Time: {0}".format(end-start))
    
    
    y_all_test = np.concatenate(y_all_test)
    #y_all_proba = np.concatenate(y_all_proba)

    
 