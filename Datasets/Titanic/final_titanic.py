import string
import numpy as np
import csv
import pandas as pd
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')
from formatting import *
from Imputation import *
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage
import math
import re

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)

#Readinig in the csv file
df_ORI = pd.read_csv('train.csv',header=0)
df_test = pd.read_csv('test.csv', header=0)
#print (df_ORI.dtypes)
df = df_ORI.copy()
df = df.fillna(np.NaN )

#print (df.describe())
#print ( df.describe(include=['O']) )
#print ("\n\n", df_test.describe() )
#print (df_test.describe(include=['O'])  )

###########################################################################################
# TRAINING DATA
###########################################################################################

df['Gender'] = df['Sex'].map( {'female': -1, 'male': 1} ).astype(int)
del df['Sex']

df_test['Gender'] = df_test['Sex'].map( {'female': -1, 'male': 1} ).astype(int)
del df_test['Sex']
print("Gender map(Sex): 'male'= 1 and 'female'= -1")


df['EmbarkIsC'] = df['Embarked'].map( lambda x: 1 if x is 'C' else -1 )
df['EmbarkIsS'] = df['Embarked'].map( lambda x: 1 if pd.isnull(x) or x is 'S' else -1 )
df['EmbarkIsQ'] = df['Embarked'].map( lambda x: 1 if x is 'Q' else -1 )
del df['Embarked']

df_test['EmbarkIsC'] = df_test['Embarked'].map( lambda x: 1 if x is 'C' else -1 )
df_test['EmbarkIsS'] = df_test['Embarked'].map( lambda x: 1 if pd.isnull(x) or x is 'S' else -1 )
df_test['EmbarkIsQ'] = df_test['Embarked'].map( lambda x: 1 if x is 'Q' else -1 )
del df_test['Embarked']
print("Embark is split into several yes/no ( 1/-1 ) columns for every letter (EmbarkIsC): C, Q, S+NaN")


df['YesMiss']   = df['Name'].map( lambda x: 1 if x.find('Miss.') != -1 or x.find('Ms.') != -1 or x.find('Mlle.') != -1 else -1)
df['YesMrs']    = df['Name'].map( lambda x: 1 if x.find('Mrs.') != -1  or x.find('Mme.')  != -1 else -1)
df['YesMr']     = df['Name'].map( lambda x: 1 if x.find('Mr.') != -1 else -1)
df['YesRev']    = df['Name'].map( lambda x: 1 if x.find('Rev.') != -1 else -1)
df['YesMaster'] = df['Name'].map( lambda x: 1 if x.find('Master.') != -1 else -1)
df['YesFancy']  = df['Name'].map( lambda x: 1 if x.find('Countess.') != -1 or x.find('Dr.') != -1 or x.find('Capt.') != -1 or x.find('Major.') != -1 or x.find('Don') != -1
                                                                   or x.find('Col.') != -1 or x.find('Sir.') != -1 or x.find('Mme.') != -1 or x.find('Lady.') != -1 or x.find('Jonkheer') != -1
                                                                   or x.find('Dona') != -1 else -1)
del df['Name']

df_test['YesMiss']   = df_test['Name'].map( lambda x: 1 if x.find('Miss.') != -1 or x.find('Ms.') != -1 or x.find('Mlle.') != -1 else -1)
df_test['YesMrs']    = df_test['Name'].map( lambda x: 1 if x.find('Mrs.') != -1  or x.find('Mme.')  != -1 else -1)
df_test['YesMr']     = df_test['Name'].map( lambda x: 1 if x.find('Mr.') != -1 else -1)
df_test['YesRev']    = df_test['Name'].map( lambda x: 1 if x.find('Rev.') != -1 else -1)
df_test['YesMaster'] = df_test['Name'].map( lambda x: 1 if x.find('Master.') != -1 else -1)
df_test['YesFancy']  = df_test['Name'].map( lambda x: 1 if x.find('Countess.') != -1 or x.find('Dr.') != -1 or x.find('Capt.') != -1 or x.find('Major.') != -1 or x.find('Don') != -1
                                                                   or x.find('Col.') != -1 or x.find('Sir.') != -1 or x.find('Mme.') != -1 or x.find('Lady.') != -1 or x.find('Jonkheer') != -1
                                                                   or x.find('Dona') != -1 else -1)
del df_test['Name']
print ("Titles Split into: Mrs.,  Mr.,  Miss.,  Rev., Master., Other")

df_noFareNaN = df[ pd.notnull(df['Fare'])]
for index, row in df.iterrows():
  if pd.isnull(row['Fare']):
    df.at[index,'Fare'] = np.mean( df_noFareNaN.loc[(df_noFareNaN['Pclass'] == row['Pclass']) & (df_noFareNaN['Gender'] == row['Gender']) 
                                                  & (df_noFareNaN['Age'] <= row['Age'] + 10) & (df_noFareNaN['Age'] <= row['Age'] - 10 ), 'Fare'] )

df.loc[(df['Fare'] <= 5), 'Fare'] = -1
df.loc[(df['Fare'] > 5) & (df['Fare'] <= 15), 'Fare'] = -.5
df.loc[(df['Fare'] > 15) & (df['Fare'] <= 40), 'Fare'] = 0
df.loc[(df['Fare'] > 40) & (df['Fare'] <= 90), 'Fare'] = .5 
df.loc[(df['Fare'] > 90), 'Fare'] = 1

df_noFareNaN = df_test[ pd.notnull(df_test['Fare'])]
for index, row in df_test.iterrows():
  if pd.isnull(row['Fare']):
    df_test.at[index,'Fare'] = np.mean( df_noFareNaN.loc[(df_noFareNaN['Pclass'] == row['Pclass']) & (df_noFareNaN['Gender'] == row['Gender'])
                                                       & (df_noFareNaN['Age'] <= row['Age'] + 10) & (df_noFareNaN['Age'] <= row['Age'] - 10 ), 'Fare'] )

df_test.loc[(df_test['Fare'] <= 5), 'Fare'] = -1
df_test.loc[(df_test['Fare'] > 5 ) & (df_test['Fare'] <= 15), 'Fare'] = -.5
df_test.loc[(df_test['Fare'] > 15) & (df_test['Fare'] <= 40), 'Fare'] = 0
df_test.loc[(df_test['Fare'] > 40) & (df_test['Fare'] <= 90), 'Fare'] = .5
df_test.loc[(df_test['Fare'] > 90), 'Fare'] = 1
print ('Fare split into: 0->5, 5->15, 10->40, 40->90, >90')


#df['HasCabin'] = df['Cabin'].map( lambda x: -1 if type(x) == float else 1)
df['Cabin'] = df['Cabin'].map(lambda x: np.NaN if pd.isnull(x) else x.rpartition(' ')[-1] )
df['CabinNumbers'] = df['Cabin'].map( lambda x: -1 if pd.isnull(x) else 0 if re.sub("\D", "", x) == ''  else int(re.sub("\D", "", x)) )
df['CabinLetters'] = df['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else re.sub(r'\d+', '', x) ) 
uniqueLetters = df.CabinLetters.unique()
for let in uniqueLetters:
  if pd.isnull(let):
    continue
  df['Cabin'+let] = df['CabinLetters'].map( lambda x: 1 if x == let else -1)
del df['Cabin']
del df['CabinLetters']

#df_test['HasCabin'] = df_test['Cabin'].map( lambda x: -1 if type(x) == float else 1)
df_test['Cabin'] = df_test['Cabin'].map(lambda x: np.NaN if pd.isnull(x) else x.rpartition(' ')[-1] )
df_test['CabinNumbers'] = df_test['Cabin'].map( lambda x: -1 if pd.isnull(x) else 0 if re.sub("\D", "", x) == ''  else int(re.sub("\D", "", x)) )
df_test['CabinLetters'] = df_test['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else re.sub(r'\d+', '', x) ) 
uniqueLetters = df_test.CabinLetters.unique()
for let in uniqueLetters:
  if pd.isnull(let):
    continue
  df_test['Cabin'+let] = df_test['CabinLetters'].map( lambda x: 1 if x == let else -1)
df_test['CabinT'] = -1
del df_test['Cabin']
del df_test['CabinLetters']
print("Cabin is split into several yes/no ( 1/-1 ) columns for every letter it starts with: C, E, G, D, A, B, T, F")


df['TicketBeginLetter'] = df['Ticket'].map( lambda x: -1 if x[0].isdigit() else 1)
df['TicketDots'] = df['Ticket'].map(lambda x: x.count("/") )
df['TicketDashes'] = df['Ticket'].map(lambda x: x.count(".") )
df['nTicketStrings'] = df['Ticket'].map(lambda x: x.count(" ")+1 )
df['TicketNumber'] = df['Ticket'].map(lambda x: 0 if re.sub("\D", "", x) == '' else int(re.sub("\D", "", x)) )
del df['Ticket']

df_test['TicketBeginLetter'] = df_test['Ticket'].map( lambda x: -1 if x[0].isdigit() else 1)
df_test['TicketDots'] = df_test['Ticket'].map(lambda x: x.count("/") )
df_test['TicketDashes'] = df_test['Ticket'].map(lambda x: x.count(".") )
df_test['nTicketStrings'] = df_test['Ticket'].map(lambda x: x.count(" ")+1 )
df_test['TicketNumber'] = df_test['Ticket'].map(lambda x: 0 if re.sub("\D", "", x) == '' else int(re.sub("\D", "", x) ) )
del df_test['Ticket']
print("TicketBeginLetter map(Ticket): 1 if Letter is in name, 0 if no letter in name.")


df['FamilySize'] =  df.fillna(0)['SibSp'] + df.fillna(0)['Parch'] + 1
df_test['FamilySize'] =  df_test.fillna(0)['SibSp'] + df_test.fillna(0)['Parch'] + 1
print("FamilySize = SibSp + Parch")

############################
# Imputing the Missing Ages
############################
from sklearn.ensemble import ExtraTreesRegressor
#paramDict = {"""'min_samples_split': [5,10,15,20],""" 'min_samples_leaf':[2,4,6,8,10]}
EXTreeReg = ExtraTreesRegressor(max_depth=6, min_samples_split=15, min_samples_leaf=4, n_estimators= 10, bootstrap=True, oob_score=True)
df_temp = impute(clf=EXTreeReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=None) #paramDict)
plt.plot(EXTreeReg.feature_importances_)
print ("Oob_Score=", EXTreeReg.oob_score_ )

"""
df_noAgeNaN = df[ pd.notnull(df['Age']) ]
for index, row in df.iterrows():
  if pd.isnull(row['Age']):
    df.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )
"""
df['AgeGroup'] = df['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df['Age']

df_temp = impute(clf=EXTreeReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=None)
"""
df_noAgeNaN = df_test[ pd.notnull(df_test['Age']) ]
for index, row in df_test.iterrows():
  if pd.isnull(row['Age']):
    df_test.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )
"""
df_test['AgeGroup'] = df_test['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df_test['Age']
print ("AgeGroup map(Age): '-1': 0 <= x < 10 | '-.5': 10 <= x < 20 | '0': 20 <= x 30 | '.5': 30 <= x < 50 | '1': x <= 50")


##################################
# Selecting features of interest
##################################
# All numerical or boolean features
df_noStrings = df.drop(['PassengerId'], axis=1)
df_noStrings = df_noStrings.fillna(df_noStrings.mean())

np_train_features_noStrings = df_noStrings.values
labels_train, features_train = targetFeatureSplit(np_train_features_noStrings )

from sklearn.model_selection import train_test_split
features_train_1, features_test_1, labels_train_1, labels_test_1 = train_test_split(features_train, labels_train, test_size=0.25, random_state=42)

np_passengerID = df_test['PassengerId'].values
df_test_noStrings = df_test.drop(['PassengerId'], axis=1)
df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mean())

features_test = df_test_noStrings.values

giniDecGen, giniDevGenTup =  calcGiniImp(df_noStrings, 'Survived', 'Gender') 
giniDecGen, giniDevGenTup =  calcGiniImp(df_noStrings, 'Survived', 'YesMr')

"""
################
# Classifying
################
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')
from sklearn.ensemble import RandomForestClassifier

#############  Training Random Forest ##############
clf1 = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=1, min_samples_split=5, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf1.fit(features_train_1, labels_train_1)
print ("clf1=", clf1.score(features_test_1, labels_test_1), "\toob_score=", clf1.oob_score_ )

clf2 = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf2.fit(features_train_1, labels_train_1)
print ("clf2=",  clf2.score(features_test_1, labels_test_1), "\toob_score=", clf2.oob_score_ )

clf3 = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_leaf=1, min_samples_split=15, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf3.fit(features_train_1, labels_train_1)
print ("clf3=", clf3.score(features_test_1, labels_test_1), "\toob_score=", clf3.oob_score_ )

clf4 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=5, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf4.fit(features_train_1, labels_train_1)
print ("clf4=", clf4.score(features_test_1, labels_test_1), "\toob_score=", clf4.oob_score_ )

clf5 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf5.fit(features_train_1, labels_train_1)
print("clf5=", clf5.score(features_test_1, labels_test_1), "\toob_score=", clf5.oob_score_ )

clf6 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=15, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf6.fit(features_train_1, labels_train_1)
print ("clf6=", clf6.score(features_test_1, labels_test_1) , "\toob_score=", clf6.oob_score_)

clf7 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=5, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf7.fit(features_train_1, labels_train_1)
print("clf7=", clf7.score(features_test_1, labels_test_1) , "\toob_score=", clf7.oob_score_) 

clf8 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf8.fit(features_train_1, labels_train_1)
print ("clf8=", clf8.score(features_test_1, labels_test_1) , "\toob_score=", clf8.oob_score_)

clf9 = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_leaf=1, min_samples_split=15, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf9.fit(features_train_1, labels_train_1)
print ("clf9=", clf9.score(features_test_1, labels_test_1) , "\toob_score=", clf9.oob_score_)

#############33 Random Forest ###################
clf = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_split=15, min_samples_leaf=1, class_weight=None, n_estimators=1000 )

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
pred = pred.astype(int)

df_answers = pd.DataFrame({'PassengerId': np_passengerID, 'Survived': pred} )
df_answers.to_csv("Answers/ExtTreeImpAge_RFC_maxDepth5_gini_minSamplesSplit10_minSamplesLeaf2_weightBalanced_nEstimators100.csv", sep=',', index=False)

############################################################################
# Showing the feature importances before and after making choices on values
############################################################################
maxLabel = findMaxCorr(df_noStrings)
newDfs = {k: v for k,v in df_noStrings.groupby(maxLabel)}

for k,dfIte in newDfs.items():
  print ("k=", k) 
  clfIte = RandomForestClassifier(criterion='entropy', max_depth=4, min_samples_split=10, min_samples_leaf=2, class_weight='balanced', n_estimators=250 )
  dfIte = dfIte.drop([maxLabel], axis=1)
  np_train_features_noStrings_ite = dfIte.values
  labels_train_ite, features_train_ite = targetFeatureSplit(np_train_features_noStrings_ite )
  clfIte.fit(features_train_ite, labels_train_ite)
  visualizeDecisionTree(dfIte)

  maxLabel_1 = findMaxCorr(dfIte)
  newDfs_1 = {k_1: v_1 for k_1,v_1 in dfIte.groupby(maxLabel_1)}
  for k_1,dfIte_1 in newDfs_1.items():
    print ("  k_1=", k_1) 
    clfIte = RandomForestClassifier(criterion='entropy', max_depth=4, min_samples_split=10, min_samples_leaf=2, class_weight='balanced', n_estimators=250 )
    dfIte_1 = dfIte_1.drop([maxLabel_1], axis=1)
    np_train_features_noStrings_ite_1 = dfIte_1.values
    labels_train_ite_1, features_train_ite_1 = targetFeatureSplit(np_train_features_noStrings_ite_1 )
    clfIte.fit(features_train_ite_1, labels_train_ite_1)
    visualizeDecisionTree(dfIte_1)

    maxLabel_2 = findMaxCorr(dfIte_1)
    newDfs_2 = {k_2: v_2 for k_2,v_2 in dfIte_1.groupby(maxLabel_2)}
    for k_2,dfIte_2 in newDfs_2.items():
      print ("    k_2=", k_2)
      clfIte = RandomForestClassifier(criterion='entropy', max_depth=4, min_samples_split=10, min_samples_leaf=2, class_weight='balanced', n_estimators=250 ) 
      dfIte_2 = dfIte_2.drop([maxLabel_2], axis=1)
      np_train_features_noStrings_ite_2 = dfIte_2.values
      labels_train_ite_2, features_train_ite_2 = targetFeatureSplit(np_train_features_noStrings_ite_2 )
      clfIte.fit(features_train_ite_2, labels_train_ite_2)
      visualizeDecisionTree(dfIte_2)
      maxLabel_3 = findMaxCorr(dfIte_2)
    


##################################
#    Visualization
##################################
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 300) # viridis
cax = ax1.imshow(df_noStrings.corr(), interpolation="nearest", cmap=cmap)
ax1.grid(True, color='grey')
plt.title('Titanic Correlation of Features', y=1.1, size=15)
labels = [column for column in df_noStrings]
ax1.set_xticks(np.arange(len(labels))-.5)
ax1.set_xticklabels(labels,fontsize=6, rotation=45, ha='right')
ax1.set_yticks(np.arange(len(labels))-.5)
ax1.set_yticklabels(labels,fontsize=6, rotation=45, va='top')
ax1.set_xlim(16.5, -.5)
ax1.set_ylim(16.5, -.5)
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[1, .75, .5, .25, 0, -.25, -.5, -.75, -1])
ite = 0
for i in df_noStrings.corr():
  jte = 0
  for j in df_noStrings.corr():
    ax1.annotate(round(df_noStrings.corr()[i][j], 2), (ite+.35,jte) )
    jte += 1
  ite += 1
plt.show()

# Export our trained model as a .dot file
from sklearn import tree
from inspect import getmembers
labels = []
for col in df_test_noStrings:
  labels.append(col)

iTree = 0
for tree_in_forest in clf.estimators_:
    with open('TreeOutput/tree_' + str(iTree) + '_max_depth5.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file=my_file, max_depth=5, impurity=True, feature_names=list(df_test_noStrings), 
                                       class_names=['Died', 'Survived'], rounded=True, filled=True)
    print ("Tree #", iTree)
    for i in range(len(labels) ):
      print ("\t", labels[i], tree_in_forest.feature_importances_[i] )
    iTree = iTree + 1
    #Convert .dot to .png to allow display in web notebook
for ite in range(iTree):
    check_call(["dot","-Tpng", 'TreeOutput/tree_' + str(ite) + '_max_depth5.dot', "-o",'TreeOutput/tree_' + str(ite) + '_max_depth5.png'])
"""
