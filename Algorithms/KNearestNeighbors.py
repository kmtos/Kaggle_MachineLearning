#######################################################
# Defining and running Nearest Neighbor Classification
#######################################################
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
import math
from formatting import *


def kNNClassifier( features=None, labels=None, kFolds=2, printScores=False, kValue=5, weights='uniform', metric='minkowski', power=2, metricParams=None):
  ###########################
  # Tree Classifier + kFolds
  ###########################
  if features is None:
    print ("Give me some data to analyze.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  kNN_kFoldsScores = []
  print ("\n############################\nk Nearest Neighbor\n kValue=", kValue, " \n weights=", weights, "\n metric=", metric, "\n power=", power, "\n metricParams=", metricParams,
	 "\n kFolds=", kFolds, "\n#############################################")

  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]

    clf = KNeighborsClassifier(n_neighbors=kValue, weights=weights, metric=metric, p=power, metric_params=metricParams)
    clf.fit(features_train, labels_train)
    pred =clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores: printTrueFalsePosNeg(labels_test, pred)
    kNN_kFoldsScores.append( (acc, precision, recall) )

  sumAcc = 0
  sumPre = 0
  sumRec = 0
  for tup in kNN_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]

  print ("#######################\nFinal Results\n######################")
  print (": Accuracy=", round(sumAcc / (kFolds*1.0), 3), "\tPrecision=", round(sumPre/(kFolds*1.0), 3), "\tRecall=", round(sumRec / (kFolds*1.0), 3))

def distanceWeight(matrix):
  nRows,nColumns = matrix.shape
  for row in range(nRows):
    for col in range(nColumns):
      matrix[row][col] = math.pow(matrix[row][col], .5)
  return matrix 

