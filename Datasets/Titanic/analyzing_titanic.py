import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)

#Readinig in the csv file
df_ORI = pd.read_csv('train.csv',header=0)
print (df_ORI.dtypes)

df = df_ORI.fillna(np.NaN )

df['Gender'] = df['Sex'].map( {'female': -1, 'male': 1} ).astype(int)
del df['Sex']
print("Gender map(Sex): 'male'= 1 and 'female'= -1")

df['EmbarkIsC'] = df['Embarked'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is 'C' else -1 )
df['EmbarkIsS'] = df['Embarked'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is 'S' else -1 )
df['EmbarkIsQ'] = df['Embarked'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is 'Q' else -1 )
print("Embark is split into several yes/no ( 1/-1 ) columns for every letter (EmbarkIsC): C, Q, S")

df['CabLetIsC'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'C' else -1 )
df['CabLetIsE'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'E' else -1 )
df['CabLetIsG'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'G' else -1 )
df['CabLetIsD'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'D' else -1 ) 
df['CabLetIsA'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'A' else -1 )
df['CabLetIsB'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'B' else -1 )
df['CabLetIsF'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'F' else -1 )
df['CabLetIsT'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x[0] == 'T' else -1 )
print("Cabin is split into several yes/no ( 1/-1 ) columns for every letter it starts with: C, E, G, D, A, B, T, F")

df['TicketBeginLetter'] = df['Ticket'].map( lambda x: np.NaN if pd.isnull(x) else -1 if x[0].isdigit() else 1)
print("TicketBeginLetter map(Ticket): 1 if Letter is in name, 0 if no letter in name.")

df['AgeGroup'] = df['Age'].map(lambda x: np.NaN if pd.isnull(x) else 0 if x >= 0 and x < 10 else 1 if x >= 10 and x < 20 else 2 if x >= 20 and x < 30 else 3 if x >= 30 and x < 50 else 4 if x >= 50 else -1)
print ("AgeGroup map(Age): '0': 0 <= x < 10 | '1': 10 <= x < 20 | '2': 20 <= x 30 | '3': 30 <= x < 50 | '4': x <= 50")

df['FamilySize'] =  df.fillna(0)['SibSp'] + df.fillna(0)['Parch'] 
print("FamilySize = SibSp + Parch")

titleDict = {}
for index, row in df.iterrows():
   if row['Name'].find('.') != -1:
     endPoint = row['Name'].find('.')
     startPoint = endPoint
     while startPoint > 0 and row['Name'][startPoint] != ' ':
       startPoint += -1
     titleDict[row['Name'][startPoint+1:endPoint+1] ] = titleDict.get(row['Name'][startPoint+1:endPoint+1], 0) + 1

print ("\n####################\nTitles and counts\n#######################")
for key, value in titleDict.items():
  print ("Title:", key ,"\tappeared", value, " times");

df['YesMiss']   = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Miss.') != -1 or x.find('Ms.') != -1 else -1)
df['YesMrs']    = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Mrs.') != -1     else -1)
df['YesMr']     = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Mr.') != -1      else -1)
df['YesRev']    = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Rev.') != -1     else -1)
df['YesMaster'] = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Master.') != -1  else -1)
df['YesFancy']  = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Countess.') != -1 or x.find('Mlle.') != -1 or 
                                            x.find('Sir.') != -1 or x.find('Mme.') != -1 or x.find('Lady.') != -1 else -1)
df['YesDr']     = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Dr.') != -1      else -1)
df['YesMillitary'] = df['Name'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x.find('Capt.') != -1 or x.find('Major.') != -1 or x.find('Col.') != -1  else -1)


##############################
# Removing rows with all NaN
##############################
print("\n\n\nBEFORE REMOVAL: len(df)=", len(df) )
df['nanSum'] = 0
for ir in df.itertuples():
  nanSum = 0
  for i in ir[1:]:
    if pd.isnull(i):
      nanSum += 1
  df.iloc[int(ir[0]), df.columns.get_loc('nanSum')] = nanSum
df = df[df['nanSum'] < len(df.columns) - 1]
print("AFTER REMOVAL: len(df)=", len(df) )

 
##################################
# Selecting features of interest
##################################
# All numerical or boolean features
df_noStrings = df.drop(['PassengerId', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1)
df_noStrings = df_noStrings.fillna(df.mean())
df_selection = df_noStrings.drop( ['AgeGroup', 'FamilySize'], axis=1)

np_train_features_selection = df_selection.values
labels_selection, features_selection = targetFeatureSplit(np_train_features_selection )

df_boolean = df_noStrings[['Survived', 'YesMiss',  'YesMrs', 'YesRev', 'YesMaster', 'YesFancy', 'YesDr', 'YesMillitary', 'EmbarkIsC', 'EmbarkIsS', 'EmbarkIsQ', 'CabLetIsC', 'CabLetIsE', 'CabLetIsG', 'CabLetIsD', 'CabLetIsA', 'CabLetIsB', 'CabLetIsF', 'CabLetIsT', 'TicketBeginLetter', 'Gender']]
np_train_features_boolean = df_boolean.values
labels_boolean, features_boolean = targetFeatureSplit(np_train_features_boolean )

df_SVM = df_noStrings[['Survived', 'YesMiss',  'YesMrs', 'YesRev', 'YesMaster', 'YesFancy', 'YesDr', 'YesMillitary', 'EmbarkIsC', 'EmbarkIsS', 'EmbarkIsQ', 'CabLetIsC', 'CabLetIsE', 'CabLetIsG', 'CabLetIsD', 'CabLetIsA', 'CabLetIsB', 'CabLetIsF', 'CabLetIsT', 'TicketBeginLetter', 'Gender', 'AgeGroup', 'FamilySize', 'Pclass', ]]
np_train_features_SVM = df_SVM.values
labels_SVM, features_SVM = targetFeatureSplit(np_train_features_SVM )

################
# Classifying
################
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')

# k-Nearest Neighbors
from KNearestNeighbors import *
"""
kNNClassifier( features=features_selection, labels=labels_selection, kFolds=10, printScores=False, kValue=5, weights='distance', metric='minkowski', power=2)
kNNClassifier( features=features_selection, labels=labels_selection, kFolds=10, printScores=False, kValue=20, weights=distanceWeight, metric='chebyshev', power=5)

# Decision Trees
from Trees import *
treeClassifier(features=features_selection, labels=labels_selection, maxDepth=5, criterion='entropy', minSamplesSplit=4, minSamplesLeaf=2, minWeightFractionLeaf=0., classWeight=None, 
	       kFolds=10, printScores=True)
randomForestClassifier(features=features_selection, labels=labels_selection, maxDepth=5, criterion='entropy', minSamplesSplit=4, minSamplesLeaf=2, minWeightFractionLeaf=0., classWeight=None,
                       kFolds=10, printScores=True, nEstimators=20) 


# SVM Classifiers

from SVM import *
svmClassifier(features=features_SVM, labels=labels_SVM, kernel='linear', C_smoothness=1, kFolds=10, printScores=True, polyDegree=1, gamma='auto', classWeight='balanced')
svmClassifier(features=features_SVM, labels=labels_SVM, kernel=customKernel_circle, C_smoothness=1, kFolds=10, printScores=True, polyDegree=3, gamma='auto', classWeight='balanced')

#Naive Bayes
from NaiveBayes import *
NBGaussianClassifier(features=features_selection, labels=labels_selection, kFolds=10, printScores=True)

#  Boosting

from Boosting import *
ada_clf1 = DecisionTreeClassifier(class_weight={0: 0.35, 1: 0.65}, max_depth=10, min_samples_leaf=3, min_weight_fraction_leaf=0, min_samples_split=5)
adaBoostClassifier(features=features_selection, labels=labels_selection, estimator=ada_clf1, nEstimators=100,  learningRate=.75, kFolds=10, printScores=False)

ada_clf2 = RandomForestClassifier(class_weight={0: 0.35, 1: 0.65}, criterion='entropy', max_depth=10, min_samples_split=4, min_samples_leaf=2, min_weight_fraction_leaf=0., n_estimators=20)
KNeighborsClassifier(metric='hamming', weights='uniform', n_neighbors=25)
adaBoostClassifier(features=features_selection, labels=labels_selection, estimator=ada_clf2, nEstimators=100,  learningRate=.75, kFolds=10, printScores=False)

ada_clf3 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=1, min_weight_fraction_leaf=0, min_samples_split=2)
adaBoostClassifier(features=features_selection, labels=labels_selection, estimator=ada_clf3, nEstimators=50,  learningRate=1.25, kFolds=10, printScores=False)
#svc = SVC()
#adaBoostClassifier(features=features_selection, labels=labels_selection, estimator=ada_clf3, nEstimators=50,  learningRate=1.25, kFolds=10, printScores=False)

#Grid Search
"""
from GridSearch import *


"""
paramDictTree = {'criterion':('entropy', 'gini'), 'max_depth':[2,5,10,25], 'min_samples_split':[8,5,2], 'min_samples_leaf':[3,2,1], 'min_weight_fraction_leaf': [0,.25,.5],
             'class_weight':[None, 'balanced', {0:.35, 1:.65}] }
tree = DecisionTreeClassifier()
print ('\n##########################################\n GridSearchCV with Decision Tree:')
treeBestParams = gridSearch(tree, features=features_selection, labels=labels_selection, parameters=paramDictTree, cross_validation=10)

paramDictRandomForest = {'criterion':('entropy', 'gini'), 'max_depth':[2,5], 'min_samples_split':[8,5], 'min_samples_leaf':[3,2], 'min_weight_fraction_leaf': [0,.25],
             'class_weight':['balanced', {0:.35, 1:.65}], 'n_estimators': [25, 50] }
randomForest = RandomForestClassifier()
print ('\n##########################################\n GridSearchCV with Random Forest:')
randomForestParams = gridSearch(randomForest, features=features_selection, labels=labels_selection, parameters=paramDictRandomForest, cross_validation=10)


paramDictSVM = {'kernel':('sigmoid', 'rbf'), 'C':[1, 10, 30, 70], 'gamma':[.05, .1, 1, 10, 30], 'class_weight':[None, 'balanced', {0:.35, 1:.65}]}
svc = SVC()
print ('\n##########################################\n GridSearchCV with SVM:')
SVMBestParams = gridSearch(svc, features=features_SVM, labels=labels_SVM, parameters=paramDictSVM, cross_validation=10)
"""

paramDictKNN = {'n_neighbors': [5,10, 25, 50, 100], 'weights':['uniform','distance', distanceWeight], 'metric':['braycurtis', 'canberra', 'hamming'], 'p':[1,2,3,4]}
knn = KNeighborsClassifier()
print ('\n##########################################\n GridSearchCV with k-Nearest Neighbors:')
kNNBestParams = gridSearch(knn, features=features_selection, labels=labels_selection, parameters=paramDictKNN, cross_validation=10)
"""
from time import time
t0 = time()
paramDictTreeAdaBoost = {'base_estimator__max_depth':[10,15, 20], 'base_estimator__min_samples_split':[8,5], 'base_estimator__min_samples_leaf':[4,2],
                         'base_estimator__min_weight_fraction_leaf': [.05,.3], 'base_estimator__class_weight':[None, 'balanced', {0:.35, 1:.65}], 
			 'n_estimators': [100, 150, 200], 'learning_rate': [.01, .1, 1] }
treeAdaBoost = DecisionTreeClassifier(criterion='entropy')
adaBoostWithTree = AdaBoostClassifier(base_estimator=treeAdaBoost)
print ('\n##########################################\n GridSearchCV with AdaBoost with Decision Tree:')
decisionTreeAdaBoostBestParams= gridSearch(adaBoostWithTree, features=features_selection, labels=labels_selection, parameters=paramDictTreeAdaBoost, cross_validation=10)
t1 = time()
print ("AdaBoosted DecisionTree time=", t1-t0)

t0 = time()
paramDictRandomForestAdaBoost = {'base_estimator__max_depth':[10, 15, 20], 'base_estimator__min_samples_split':[8,2], 
		         'base_estimator__min_weight_fraction_leaf': [.05,.3], 'base_estimator__class_weight':[None, 'balanced', {0:.35, 1:.65}],
                         'n_estimators': [100, 150 ], 'learning_rate': [.01, .1, 1] }
randomForestAdaBoost = RandomForestClassifier(criterion='entropy', n_estimators=50)
adaBoostWithRandomForest = AdaBoostClassifier(base_estimator= randomForestAdaBoost)
print ('\n##########################################\n GridSearchCV with AdaBoost Random Forest:')
randomForestAdaBoostBestParams = gridSearch(adaBoostWithRandomForest, features=features_selection, labels=labels_selection, parameters=paramDictRandomForestAdaBoost, cross_validation=10)
t1 = time()
print ("AdaBoosted RandomForest time=", t1-t0)
"""
################
# Visualization
################
pairs = []
pairs.append(['AgeGroup', 'Gender'])
pairs.append(['FamilySize', 'Gender'])
pairs.append(['CabLetter', 'Gender'])
pairs.append(['TicketBeginLetter', 'Gender'])
pairs.append(['Fare', 'Gender'])
pairs.append(['Pclass', 'Gender'])
pairs.append(['Destination', 'Gender'])
pairs.append(['FamilySize', 'AgeGroup'])
pairs.append(['Fare', 'Pclass'])
plotGridOf2DsWithColor(df_selection, pairs, ["blue", "red"], [ "o", "s"], "SVM_FeatureComb.png", "png" , 0.01, 110, 1)

# Testing family member 
"""
#################
# Percentages
#################
cFamilySizeGender = []
cCabLetterGender = []
cTicketBeginLetterGender = []
cFareGender = []
cPclassGender = []
cDestinationGender = []

#print df_train_features[ (df_train_features['Gender'] == 1) & (df_train_features['Gender' == 1]) ] )
for i in df_train_features['FamilySize'].unique():
  cFamilySizeGender.append( (i, 0, df_train_features[(df_train_features['FamilySize'] == i) & (df_train_features['Gender'] == 0)].count()[0], df_train_features[(df_train_features['FamilySize'] == i) & (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1)].count()[0] )  )
  cFamilySizeGender.append( (i, 1, df_train_features[(df_train_features['FamilySize'] == i) & (df_train_features['Gender'] == 1)].count()[0], df_train_features[(df_train_features['FamilySize'] == i) & (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1)].count()[0] )  )
print("cFamilySizeGender=",cFamilySizeGender )

for i in df_train_features['CabLetter'].unique():
  cCabLetterGender.append( (i, 0, df_train_features[(df_train_features['CabLetter'] == i) & (df_train_features['Gender'] == 0)].count()[0], df_train_features[(df_train_features['CabLetter'] == i) & (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1)].count()[0] )  )
  cCabLetterGender.append( (i, 1, df_train_features[(df_train_features['CabLetter'] == i) & (df_train_features['Gender'] == 1)].count()[0], df_train_features[(df_train_features['CabLetter'] == i) & (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1)].count()[0] )  )
print("\ncCabLetterGender=",cCabLetterGender )

for i in df_train_features['TicketBeginLetter'].unique():
  cTicketBeginLetterGender.append( (i, 0, df_train_features[(df_train_features['TicketBeginLetter'] == i) & (df_train_features['Gender'] == 0)].count()[0], df_train_features[(df_train_features['TicketBeginLetter'] == i) & (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1)].count()[0] )  )
  cTicketBeginLetterGender.append( (i, 1, df_train_features[(df_train_features['TicketBeginLetter'] == i) & (df_train_features['Gender'] == 1)].count()[0], df_train_features[(df_train_features['TicketBeginLetter'] == i) & (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1)].count()[0] )  )
print("\ncTicketBeginLetterGender=", cTicketBeginLetterGender )

for i in df_train_features['Pclass'].unique():
  cPclassGender.append( (i, 0, df_train_features[(df_train_features['Pclass'] == i) & (df_train_features['Gender'] == 0)].count()[0], df_train_features[(df_train_features['Pclass'] == i) & (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1)].count()[0] )  )
  cPclassGender.append( (i, 1, df_train_features[(df_train_features['Pclass'] == i) & (df_train_features['Gender'] == 1)].count()[0], df_train_features[(df_train_features['Pclass'] == i) & (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1)].count()[0] )  )
print("\ncPclassGender=", cPclassGender )

for i in df_train_features['Destination'].unique():
  cDestinationGender.append( (i, 0, df_train_features[(df_train_features['Destination'] == i) & (df_train_features['Gender'] == 0)].count()[0], df_train_features[(df_train_features['Destination'] == i) & (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1)].count()[0] )  )
  cDestinationGender.append( (i, 1, df_train_features[(df_train_features['Destination'] == i) & (df_train_features['Gender'] == 1)].count()[0], df_train_features[(df_train_features['Destination'] == i) & (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1)].count()[0] )  )
print("\ncDestinationGender=", cDestinationGender )


print("Gender=", 0, "\tPercentage=", df_train_features[ (df_train_features['Gender'] == 0) & (df_train_labels['Survived'] == 1) ].count()[0] / (1.0 * df_train_features[ (df_train_features['Gender'] == 0) ].count()[0]) )
print("Gender=", 1, "\tPercentage=", df_train_features[ (df_train_features['Gender'] == 1) & (df_train_labels['Survived'] == 1) ].count()[0] / (1.0 * df_train_features[ (df_train_features['Gender'] == 1) ].count()[0]) )

for i in cCabLetterGender:
  print("CabLetter=",i[0], "\tGender=",  i[1], "\tPercent=", i[3] / (i[2] * 1.0), "\ti=", i )
  

for i in cFamilySizeGender:
  print("FamilySize=",i[0], "\tGender=",  i[1], "\tPercent=", i[3] / (i[2] * 1.0), "\ti=", i )

for i in cTicketBeginLetterGender:
  print("TicketBeginLetter=", i[0], "\tGender=", i[1], "\tPercent=", i[3] / (i[2] * 1.0), "\ti=", i ) 

for i in cPclassGender:
  print("Pclass=", i[0], "\tGender=", i[1], "\tPercent=", i[3] / (i[2] * 1.0), "\ti=", i )

for i in cDestinationGender:
  print("Destination=", i[0], "\tGender=",  i[1], "\tPercent=", i[3] / (i[2] * 1.0), "\ti=", i )

"""
