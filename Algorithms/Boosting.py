##################################################
# Defining and running Decision Tree Classifiers
##################################################
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import defaultdict
import sys
from sklearn.ensemble import AdaBoostClassifier
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *

def adaBoostClassifier(features=None, labels=None, estimator=None, nEstimators=50, learningRate=1., kFolds=2, printScores=False):
  ###########################
  # Tree Classifier + kFolds
  ###########################
  if estimator is None or features is None or labels is None:
    print ("Give me some data to analyze, labels to look at, or an estimator to run with.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  tree_kFoldsScores = []
  print ("\n####################################\nAdaBoost With", estimator, "\n  n_estimator=", nEstimators, "\n  learningRate=", learningRate, "\n  kFolds=", kFolds, "\n###########################")

  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]

    clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=nEstimators, learning_rate=learningRate)
    clf.fit(features_train, labels_train)
    pred =clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores: printTrueFalsePosNeg(labels_test, pred)
    tree_kFoldsScores.append( (acc, precision, recall) )

  sumAcc = 0
  sumPre = 0
  sumRec = 0
  for tup in tree_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]

  print ("#######################\nFinal Results\n######################")
  print ("Accuracy=", round(sumAcc / (kFolds*1.0), 3), "\tPrecision=", round(sumPre/(kFolds*1.0), 3), "\tRecall=", round(sumRec / (kFolds*1.0), 3))



