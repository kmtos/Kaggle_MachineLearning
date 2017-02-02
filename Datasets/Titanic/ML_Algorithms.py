  
  
  
def svmClassifier( features=None, labels=None, kernel="linear", C_smoothness=1, kFolds=1, printScores=False, polyDegree=3, gamma=auto, classWeight=None)  
  ##########################
  # SVM Classifier + kFolds
  ##########################
  if features is None:
    print ("Give me some data to analyze.")
    return
  from sklearn.svm import SVC
  from sklearn import metrics
  from sklearn.model_selection import KFold
  
  kF = KFold( n_splits=kFolds, shuffle=True)
  svm_kFoldsScores = []
  svmRBF_kFoldsScores = []
  svmPoly_kFoldsScores = defaultdict(list)
  print ("\n##########################################\n", kernel,"  SVM Fit\n\tC=", C_smoothness, "\tgamma=", gamma, "\tdegree=", polyDegree, "\tclass_weight=", classWeight,
         "\n\nkFolds=", kFolds, "\n##################################")
  for kFolds_train_index, kFolds_test_index in kF.split(labels):
    features_train = [features_selection[i] for i in kFolds_train_index]
    labels_train   = [labels_selection[i]   for i in kFolds_train_index]
    features_test  = [features_selection[i] for i in kFolds_test_index]
    labels_test    = [labels_selection[i]   for i in kFolds_test_index]
  
    clf = SVC(kernel="linear", C=C_smoothness, degree=polyDegree, gamma=gamma, class_weight=classWeight)
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
  
  print ("\n\n####################### Final Results\n ######################")
  print (kernel, ": Accuracy=", round(sumAcc / 10.0, 3), "\tPrecision=", round(sumPre/10.0, 3), "\tRecall=", round(sumRec / 10.0, 3))


def svmClassifierGridSearch( features=None, labels=None, parameters=None, cross_validation=5)  
  ###############################
  # SVM Classifier GridSearchCV
  ##############################
  if features is None:
    print ("Give me some data")
    return
  if parameters is None:
    print ("Give me some parameters to iterate over.")
    return
  from sklearn.svm import SVC
  from sklearn import metrics
  from sklearn.model_selection import GridSearchCV
  
  print ("\n##########################################\n SVM Fit with GridSearchCV:")
  for key,value in parameters:
    print ("\t", key, "=", values)
  print ("\n##################################")
  clf = GridSearchCV( estimator=svc, parameters, cv=cross_validation)
  clf.fit(features_train, labels_train)
  
  print ("BestScore=", clf.best_score_, "\nbest Parameters=", clf.beist_params_)
  
