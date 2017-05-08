##########################
# Imputing missing values
##########################
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.base import clone
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *
import math
from GridSearch import *
from sklearn.model_selection import train_test_split

#############################################
# Make sure predictive label isn't included.
# The label is the imputed variable
#############################################
def impute(clf=None, df=None, colName=None, isDiscrete=True, colsToRemovePrior=[], paramDict=None):
  if clf is None or df is None or colName is None: 
    print ("Give me features, labels and clf!!!!")
    return 0

  # Turning the reassigning of values of a copy off
  pd.options.mode.chained_assignment = None

  # Moving imputed column to front
  cols = list(df)
  cols.insert(0, cols.pop( cols.index(colName) ) )
  df_moved = df.ix[:, cols]
  listTempRemovedCols = []
  for col in colsToRemovePrior:
    if col in df_moved:
      df_moved = df_moved.drop( [col], axis=1)

  df_train = df_moved[pd.notnull(df_moved[colName] )]
  df_test  = df_moved[pd.isnull(df_moved[colName] )]
  df_test  = df_test.drop([colName], axis=1)

  for col in df_train:
    if col != colName:
      df_train[col] = df_train[col].fillna(df_train[col].mean() )
      df_test[col]  = df_test[col ].fillna(df_test[col ].mean() )

  # Removing other columns with NaN's after the value was replaced with means
  listNaNCols = df_train.columns[pd.isnull(df_train).sum() > 0].tolist()
  listNaNCols =  df_test.columns[pd.isnull(df_test ).sum() > 0].tolist() + listNaNCols
  listNaNCols = [x for x in listNaNCols if x != colName]
  for col in listNaNCols:
    df_train = df_train.drop([col], axis=1)
    df_test = df_test.drop([col], axis=1)
  print("Columns remobed due to NaN's=", listNaNCols )

  np_train_features = df_train.values
  labels_train, features_train = targetFeatureSplit(np_train_features )
  features_train, features_test, labels_train, labels_test = train_test_split(features_train, labels_train, test_size=0.33, random_state=42)

  if paramDict is not None:
    clfClone = clone(clf)
    bestParams = gridSearch(clfClone, features=features_train, labels=labels_train, parameters=paramDict, cross_validation=5)
    clf = clf.set_params(**bestParams)


  labels_train, features_train = targetFeatureSplit(np_train_features )
  clf.fit(features_train, labels_train)
  for index,row in df_test.iterrows():
    features_test = np.array([row.values])
    pred = clf.predict(features_test)
    df.set_value(index, 'Age', pred)

  return df
