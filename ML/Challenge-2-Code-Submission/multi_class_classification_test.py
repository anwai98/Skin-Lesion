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

# Classifiers
#====================================================================
Classifiers = [
               
               # svm.LinearSVC(), #*
               # LogisticRegression(C=1, multi_class='multinomial'), #*
               # RidgeClassifier(), #*
               svm.SVC(kernel='rbf', C=10, probability=True, gamma = 0.0030), # too slow #*
               # RandomForestClassifier(random_state=0, criterion='gini')

               ]

# Validation Dataset 
#====================================================================
for classifier in Classifiers:

    dt_df = dt.fread('csv_files/xxx_features-GLCM-C2.csv')
    df1 = dt_df.to_pandas()

    imageID = np.asarray(df1.iloc[:, 0])
    #print(imageID)
    
    dt_df = dt.fread('csv_files/xxx_features-LBP-C2.csv')
    df2 = dt_df.to_pandas()

    dt_df = dt.fread('csv_files/xxx_features-color-C2.csv')
    df3 = dt_df.to_pandas()
    
    train1_data = np.asarray(df1.iloc[:, 1:])
    train2_data = np.asarray(df2.iloc[:, 1:])
    train3_data = np.asarray(df3.iloc[:, 1:])

    X_val = np.concatenate((train1_data, train2_data, train3_data), axis=1)
    print(X_val.shape)

    print('____________________________________________')
    print('Classifier: '+classifier.__class__.__name__)

    filename = classifier.__class__.__name__ + ".sav"
    [scaler, model, sel] = pickle.load(open(filename, 'rb'))
    
    # Scaling the data
    #====================================================================
    X_val = scaler.transform(X_val)
    X_val = sel.transform(X_val)

    y_pred_val = model.predict(X_val)
    
    with open('Challenge2-Predictions.csv', 'a+') as csvfile:
        for i in range(0,len(y_pred_val),1):
            csvfile.write("{0},{1}\n".format(imageID[i], y_pred_val[i]))
    
    print("\n\n")

