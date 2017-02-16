###############################################
# Defining and running Naive Bayes Classifiers
###############################################
from sklearn.model_selection import KFold
import numpy as np
from sklearn import metrics
import sys
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *

def NBGaussianClassifier(features=None, labels=None, kFolds=2, printScores=False):
  from sklearn.naive_bayes import GaussianNB
  if features is None:
    print ("Give me some data to analyze.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  NB_kFoldsScores = []
  print ("\n##########################################\nNaive-Bayes Gaussian Classifier\n##########################################")

  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]
    clf = GaussianNB()
    clf.fit(features, labels)
    pred =clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores: printTrueFalsePosNeg(labels_test, pred)
    NB_kFoldsScores.append( (acc, precision, recall) )
      
  sumAcc = 0
  sumPre = 0
  sumRec = 0
  for tup in NB_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]
    
  print ("#######################\nFinal Results\n######################")
  print ("\tNBGaussian: Accuracy=", round(sumAcc / (kFolds*1.0), 3), "\tPrecision=", round(sumPre/(kFolds*1.0), 3), "\tRecall=", round(sumRec / (kFolds*1.0), 3))



def NBMultinomialClassifier(features=None, labels=None, alpha=1.0, kFolds=2, printScores=False):
  from sklearn.naive_bayes import MultinomialNB

  if features is None:
    print ("Give me some data to analyze.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  NB_kFoldsScores = []
  print ("\n##########################################\nNaive-Bayes Multinomial Classifier\n##########################################")

  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]
    clf = MultinomialNB(alpha=alpha)
    clf.fit(features, labels)
    pred =clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores: printTrueFalsePosNeg(labels_test, pred)
    NB_kFoldsScores.append( (acc, precision, recall) )
      
  sumAcc = 0
  sumPre = 0 
  sumRec = 0
  for tup in NB_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]
    
  print ("#######################\nFinal Results\n######################")
  print ("\tNBGaussian: Accuracy=", round(sumAcc / (kFolds*1.0), 3), "\tPrecision=", round(sumPre/(kFolds*1.0), 3), "\tRecall=", round(sumRec / (kFolds*1.0), 3))




