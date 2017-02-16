##################################################
# Defining and running Decision Tree Classifiers
##################################################
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *


def treeClassifier( features=None, labels=None, maxDepth=None, criterion='gini', minSamplesSplit=2, minSamplesLeaf=1, minWeightFractionLeaf=0., classWeight=None, kFolds=2, printScores=False):
  ###########################
  # Tree Classifier + kFolds
  ###########################
  if features is None:
    print ("Give me some data to analyze.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  tree_kFoldsScores = []
  print ("\n##########################################\n", criterion ,"Decision Tree Fit\n maxDepth=", maxDepth, " \n minSamplesSplit=", minSamplesSplit, "\n minSamplesLeaf=", minSamplesLeaf, 
	 "\n minWeightFractionLeaf=", minWeightFractionLeaf, "\n classWeight=", classWeight, "\n kFolds=", kFolds, "\n##################################")

  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]

    clf = DecisionTreeClassifier(criterion=criterion, max_depth=maxDepth, min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf, 
				 min_weight_fraction_leaf=minWeightFractionLeaf, class_weight=classWeight)
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
  print (criterion, ": Accuracy=", round(sumAcc / (kFolds*1.0), 3), "\tPrecision=", round(sumPre/(kFolds*1.0), 3), "\tRecall=", round(sumRec / (kFolds*1.0), 3))
