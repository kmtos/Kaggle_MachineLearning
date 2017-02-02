#######################################
# Defining and running SVM Classifiers
#######################################
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold
from collections import defaultdict
from sklearn.model_selection import GridSearchCV

def customKernel_circle(X, Y):
  return (np.dot(X, Y.transpose()) * np.dot(X, Y.transpose()))
  
def svmClassifier( features=None, labels=None, kernel='linear', C_smoothness=1, kFolds=1, printScores=False, polyDegree=3, gamma='auto', classWeight=None):
  ##########################
  # SVM Classifier + kFolds
  ##########################
  if features is None:
    print ("Give me some data to analyze.")
    return
  kF = KFold( n_splits=kFolds, shuffle=True)
  svm_kFoldsScores = []
  svmRBF_kFoldsScores = []
  svmPoly_kFoldsScores = defaultdict(list)
  print ("\n##########################################\n", kernel,"  SVM Fit\n C=",C_smoothness, "\n gamma=", gamma, "\n degree=", polyDegree, "\n class_weight=", classWeight,
         "\n kFolds=", kFolds, "\n##################################")
  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features[i] for i in kFolds_train_index]
    labels_train   = [labels[i]   for i in kFolds_train_index]
    features_test  = [features[i] for i in kFolds_test_index]
    labels_test    = [labels[i]   for i in kFolds_test_index]
 
    clf = SVC(kernel=kernel, C=C_smoothness, degree=polyDegree, gamma=gamma, class_weight=classWeight)
    clf.fit(features_train, labels_train)
    pred =clf.predict(features_test)
    acc = metrics.accuracy_score(pred, labels_test)
    precision = metrics.precision_score(labels_test, pred)
    recall = metrics.recall_score(labels_test, pred)
    if printScores:
      sumTruePositives = 0
      sumTrueNegatives = 0
      sumFalseNegative = 0
      sumFalsePositve = 0
      for i in range(len(pred)):
        iTrue = labels_test[i]
        iPred = pred[i]
        if iPred == iTrue and iPred == 1:
          sumTruePositives += 1
        if iPred == iTrue and iPred == 0:
          sumTrueNegatives += 1
        if iPred != iTrue and iTrue == 1:
          sumFalseNegative += 1
        if iPred != iTrue and iPred == 1:
          sumFalsePositve += 1
      print ("sumTruePositives= " , sumTruePositives , "\tsumTrueNegatives= " , sumTrueNegatives, "\tsumFalseNegative= " , sumFalseNegative , "\tsumFalsePositve= " , sumFalsePositve)
    svm_kFoldsScores.append( (acc, precision, recall) )
  
  sumAcc = 0
  sumPre = 0
  sumRec = 0
  for tup in svm_kFoldsScores:
    sumAcc += tup[0]
    sumPre += tup[1]
    sumRec += tup[2]
  
  print ("\n\n#######################\nFinal Results\n######################")
  print (kernel, ": Accuracy=", round(sumAcc / 10.0, 3), "\tPrecision=", round(sumPre/10.0, 3), "\tRecall=", round(sumRec / 10.0, 3))


def svmClassifierGridSearch( features=None, labels=None, parameters=None, cross_validation=5):  
  ###############################
  # SVM Classifier GridSearchCV
  ##############################
  if features is None:
    print ("Give me some data")
    return
  if parameters is None:
    print ("Give me some parameters to iterate over.")
    return
  print ("\n##########################################\n SVM Fit with GridSearchCV:")
  for key,value in parameters.items():
    print ("\t", key, "=", value)
  print ("##################################")
  svc = SVC()
  clf = GridSearchCV( estimator=svc, param_grid=parameters, cv=cross_validation)
  clf.fit(features, labels)
  
  print ("BestScore=", clf.best_score_, "\nbest Parameters=", clf.best_params_)
  return clf.best_params_ 
