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
print (df_ORI.dtypes ) 
df = df_ORI.fillna('-1')
print("NaN = -1")


df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
del df['Sex']
print("Gender map(Sex): 'male'= 1 and 'female'= 0")

df['Destination'] = df['Embarked'].map( {'S': 0, 'C':1, 'Q': 2, '-1': -1} ).astype(int)
del df['Embarked']
print("Destinaion map(Embarked): 'NaN' = -1,'S'=0, 'C'=1 , 'Q'=2 ")

df['CabLet'] = df['Cabin'].map( lambda x: x[0])
df['CabLetter'] = df['CabLet'].map( { '-': -1, 'C': 0, 'E': 1, 'G': 2, 'D': 3, 'A': 4, 'B': 5, 'F': 6, 'T': 7}).astype(int)
del df['Cabin']
del df['CabLet']
print("CabLetter map(Cabin)= 'NaN'= -1, 'C'= 0, 'E'= 1, 'G'= 2, 'D'= 3, 'A'= 4, 'B'= 5, 'F'= 6, 'T'= 7")

df['TicketBeginLetter'] = df['Ticket'].map( lambda x: -1 if x == -1 else 1 if x[0].isdigit() else 0)
del df['Ticket']
print("TicketBeginLetter map(Ticket): 1 if Letter is in name, 0 if no letter in name.")

df['AgeGroup'] = df['Age'].map(lambda x: -1 if x == '-1' else 0 if x >= 0 and x < 10 else 1 if x >= 10 and x < 20 else 2 if x >= 20 and x < 30 else 3 if x >= 30 and x < 50 else 4 if x >= 50 else -1)
del df['Age']
print ("AgeGroup map(Age): '0': 0 <= x < 10 | '1': 10 <= x < 20 | '2': 20 <= x 30 | '3': 30 <= x < 50 | '4': x <= 50")

df['FamilySize'] = df['SibSp'] + df['Parch']
del df['SibSp']
del df['Parch']
print("FamilySize = SibSp + Parch")

##################################
# Selecting features of interest
##################################
# All numerical or boolean features
print("\n\n#################### All Features ########################################\n" , df.columns.values )
df_train_features = df[['Survived', 'Gender', 'AgeGroup', 'FamilySize', 'CabLetter', 'Destination', 'Pclass', 'Fare', 'TicketBeginLetter']].copy()
print("\n\n################################# Total interesting features ######################\n" , df_train_features.columns.values )

#Portion of features that I use in final selection
df_train_features_selection = df[['Survived','Gender', 'Pclass', 'CabLetter', 'AgeGroup']].copy()
print("\n\n########################## Selected interesting features ###################################3\n" , df_train_features_selection.columns.values )
np_train_features_selection = df_train_features_selection.values
labels_selection, features_selection = targetFeatureSplit(np_train_features_selection )

################
# Classifying
################
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')

# k-Nearest Neighbors
from KNearestNeighbors import *
kNNClassifier( features=features_selection, labels=labels_selection, kFolds=10, printScores=False, kValue=5, weights='distance', metric='minkowski', power=2)
kNNClassifier( features=features_selection, labels=labels_selection, kFolds=10, printScores=False, kValue=20, weights=distanceWeight, metric='chebyshev', power=5)

# Decision Trees
from Trees import *
treeClassifier(features=features_selection, labels=labels_selection, maxDepth=5, criterion='entropy', minSamplesSplit=4, minSamplesLeaf=2, minWeightFractionLeaf=0., classWeight=None, 
	       kFolds=10, printScores=True)

# SVM Classifiers
from SVM import *
svmClassifier(features=features_selection, labels=labels_selection, kernel='linear', C_smoothness=1, kFolds=10, printScores=True, polyDegree=1, gamma='auto', classWeight='balanced')
svmClassifier(features=features_selection, labels=labels_selection, kernel=customKernel_circle, C_smoothness=1, kFolds=10, printScores=True, polyDegree=3, gamma='auto', classWeight='balanced')

#Naive Bayes
from NaiveBayes import *
NBGaussianClassifier(features=features_selection, labels=labels_selection, kFolds=10, printScores=True)

#Grid Search
from GridSearch import *
paramDictTree = {'criterion':('entropy', 'gini'), 'max_depth':[2,5,10,25], 'min_samples_split':[8,5,2], 'min_samples_leaf':[3,2,1], 'min_weight_fraction_leaf': [0,.25,.5],
             'class_weight':[None, 'balanced', {0:.35, 1:.65}] }
tree = DecisionTreeClassifier()
print ('\n##########################################\n GridSearchCV with Decision Tree:')
treeBestParams = gridSearch(tree, features=features_selection, labels=labels_selection, parameters=paramDictTree, cross_validation=5)

paramDictSVM = {'kernel':('sigmoid', 'rbf'), 'C':[1, 10, 30, 70], 'gamma':[.05, .1, 1, 10, 30], 'class_weight':[None, 'balanced', {0:.35, 1:.65}]}
svc = SVC()
print ('\n##########################################\n GridSearchCV with SVM:')
SVMBestParams = gridSearch(svc, features=features_selection, labels=labels_selection, parameters=paramDictSVM, cross_validation=10)

paramDictKNN = {'n_neighbors': [5,10, 25, 50, 100], 'weights':['uniform','distance', distanceWeight], 'metric':['braycurtis', 'canberra', 'hamming'], 'p':[1,2,3,4]}
knn = KNeighborsClassifier()
print ('\n##########################################\n GridSearchCV with k-Nearest Neighbors:')
kNNBestParams = gridSearch(knn, features=features_selection, labels=labels_selection, parameters=paramDictKNN, cross_validation=10)
"""
##############################
# Removing rows with all NaN
##############################
print("\n\n\nBEFORE REMOVAL: len(df_train_features)=", len(df_train_features) )
df_train_features['nanSum'] = 0
for ir in df_train_features.itertuples():
  nanSum = 0
  for i in ir[1:]:
    if i == '-1' or i == -1:
      nanSum += 1
  df_train_features.iloc[int(ir[0]), df_train_features.columns.get_loc('nanSum')] = nanSum
df_train_features = df_train_features[df_train_features['nanSum'] < len(df_train_features.columns) / 2]
print("AFTER REMOVAL: len(df_train_features)=", len(df_train_features) )
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
plotGridOf2DsWithColor(df_train_features, pairs, ["blue", "red"], [ "o", "s"], "SVM_FeatureComb.png", "png" , 0.01, 110, 1)

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
