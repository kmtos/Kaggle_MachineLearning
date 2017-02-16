import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *

def customKernel_circle(X, Y):
    return np.dot(X, Y.transpose()) * np.dot(X, Y.transpose())


def svmClassifier(features = None, labels = None, kernel = 'linear', C_smoothness = 1, kFolds = 1, printScores = False, polyDegree = 3, gamma = 'auto', classWeight = None):
  if features is None:
    print('Give me some data to analyze.')
    return None
  kF = KFold(n_splits=kFolds, shuffle=True)
  svm_kFoldsScores = []
  print('\n##########################################\n', kernel, '  SVM Fit\n C=', C_smoothness, '\n gamma=', gamma, '\n degree=', polyDegree, '\n class_weight=', classWeight, '\n kFolds=', kFolds, '\n##################################')
  for (kFolds_train_index, kFolds_test_index) in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]

    clf = SVC(kernel = kernel, C = C_smoothness, degree = polyDegree, gamma = gamma, class_weight = classWeight)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores: printTrueFalsePosNeg(labels_test, pred)
    svm_kFoldsScores.append((acc, precision, recall))
    
  sumAcc = 0
  sumPre = 0
  sumRec = 0
  for tup in svm_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]
    
  print('#######################\nFinal Results\n######################')
  print(kernel, ': Accuracy=', round(sumAcc / kFolds * 1, 3), '\tPrecision=', round(sumPre / kFolds * 1, 3), '\tRecall=', round(sumRec / kFolds * 1, 3))

