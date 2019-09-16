import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from  sklearn.tree  import  DecisionTreeClassifier
from  sklearn.ensemble  import  RandomForestClassifier , VotingClassifier
from  sklearn.linear_model  import  LogisticRegression

from  sklearn.metrics  import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report, roc_auc_score
from  sklearn.preprocessing  import  LabelEncoder , MinMaxScaler
from  sklearn  import svm #SVC , LinearSVC
from  sklearn.svm import LinearSVC
from  sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from scipy import interp

from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf


"Normalize training and testing sets"
def normalize_data(train_X, test_X, scale):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X

def train (X_tr, Y_tr, X_te, Y_te, alg):
    X_tr, X_te = normalize_data(X_tr, X_te, "minmax")
    if Y_tr.shape[1] > 1:
        Y_tr = np.argmax(Y_tr, axis=1)
        Y_te = np.argmax(Y_te, axis=1)
        
    if (alg == "svm"):
        clf = LinearSVC(random_state = 0)#0.0001, 0.001, 0.01, 0.1, 1.0
    elif (alg == "dt1"):
        clf = DecisionTreeClassifier(random_state = 0)
    elif (alg == "rf12"):
        clf = RandomForestClassifier(n_estimators=100, max_depth=80,random_state=0)
    elif (alg == "lr"):
        clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    elif (alg == "pct"): #perceptron
        clf = Perceptron(tol=1e-3, random_state=0)
    elif (alg == "nct"):
        clf = NearestCentroid()#cosine, euclidean, manhattan, mahalanobis, chebyshev
    clf.fit(X_tr, Y_tr)
    start = time.time()
    y_pred = clf.predict(X_te)
    end = time.time()
    elapsed = (end - start)/float (len(X_te))
    acc= accuracy_score(Y_te, y_pred)
    fpr_vot , tpr_vot , _ = roc_curve(Y_te , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    TN, FP, FN, TP = confusion_matrix(Y_te, y_pred).ravel()
    FalseAlarmRate = FP / float (FP + TN)
    MissRate = FN / float(TP + FN)
    print (alg)
    print('The auc: {} false alarm rate: {}  miss rate {} '.format(roc_auc_vot,FalseAlarmRate,MissRate))
    return roc_auc_vot,elapsed, FalseAlarmRate, MissRate, TN, FN