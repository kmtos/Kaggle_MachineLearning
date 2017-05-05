import string
import numpy as np
import csv
import pandas as pd
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')
from Imputation import * 

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)

#Readinig in the csv file
df_ORI = pd.read_csv('train.csv',header=0)
df_test = pd.read_csv('test.csv', header=0)
#print (df_ORI.dtypes)
df = df_ORI.copy()
df = df.fillna(np.NaN )

df['Gender'] = df['Sex'].map( {'female': -1, 'male': 1} ).astype(int)
del df['Sex']
print("Gender map(Sex): 'male'= 1 and 'female'= -1")

df['EmbarkIsC'] = df['Embarked'].map( lambda x: 1 if x is 'C' else -1 )
df['EmbarkIsS'] = df['Embarked'].map( lambda x: 1 if x is 'S' else -1 )
df['EmbarkIsQ'] = df['Embarked'].map( lambda x: 1 if x is 'Q' else -1 )
del df['Embarked']
print("Embark is split into several yes/no ( 1/-1 ) columns for every letter (EmbarkIsC): C, Q, S")

df['TicketBeginLetter'] = df['Ticket'].map( lambda x: -1 if x[0].isdigit() else 1)
del df['Ticket']
print("TicketBeginLetter map(Ticket): 1 if Letter is in name, 0 if no letter in name.")

df['FamilySize'] =  df.fillna(0)['SibSp'] + df.fillna(0)['Parch'] + 1
del df['SibSp'] 
del df['Parch'] 
print("FamilySize = SibSp + Parch")

df['YesMiss']   = df['Name'].map( lambda x: 1 if x.find('Miss.') != -1 or x.find('Ms.') != -1 or x.find('Mlle.') != -1 else -1)
df['YesMrs']    = df['Name'].map( lambda x: 1 if x.find('Mrs.') != -1  or x.find('Mme.')  != -1 else -1)
df['YesMr']     = df['Name'].map( lambda x: 1 if x.find('Mr.') != -1 else -1)
df['YesRev']    = df['Name'].map( lambda x: 1 if x.find('Rev.') != -1 else -1)
df['YesMaster'] = df['Name'].map( lambda x: 1 if x.find('Master.') != -1 else -1)
df['YesFancy']  = df['Name'].map( lambda x: 1 if x.find('Countess.') != -1 or x.find('Dr.') != -1 or x.find('Capt.') != -1 or x.find('Major.') != -1 or x.find('Don') != -1
                                                                   or x.find('Col.') != -1 or x.find('Sir.') != -1 or x.find('Mme.') != -1 or x.find('Lady.') != -1 or x.find('Jonkheer') != -1
                                                                   or x.find('Dona') != -1 else -1)
del df['Name']
print ("Titles Split into: Mrs.,  Mr.,  Miss.,  Rev., Master., Other")

df.loc[(df['Fare'] <= 5), 'Fare'] = -1
df.loc[(df['Fare'] > 5) & (df['Fare'] <= 10), 'Fare'] = -.6
df.loc[(df['Fare'] > 10) & (df['Fare'] <= 25), 'Fare'] = -.2
df.loc[(df['Fare'] > 25) & (df['Fare'] <= 50), 'Fare'] = .2
df.loc[(df['Fare'] > 50) & (df['Fare'] <= 100), 'Fare'] = .6
df.loc[(df['Fare'] > 100), 'Fare'] = 1
print ('Fare split into: 0->5, 5->10, 10->25, 25->50, 50->100, >100 centered around 0.')

df['HasCabin'] = df['Cabin'].map( lambda x: -1 if type(x) == float else 1)
del df['Cabin']



# Imputing missing values
"""
#from sklearn import linear_model
#LassReg = linear_model.Lasso()
#paramDict={'alpha': [.1, 1, 2, 5, 10], 'max_iter': [1000,10000,20000], 'tol': [.01, .001, .0001, .00001]}
#df_temp = impute(clf=LassReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=paramDict)

#from sklearn.neighbors import KNeighborsRegressor
#paramDict = {'n_neighbors': [3,5,10,15,20], 'weights': [None, 'distance'], 'metric':['braycurtis', 'canberra', 'hamming'] }
#KNNReg = KNeighborsRegressor()
#df_temp = impute(clf=KNNReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=paramDict)

#from sklearn.ensemble import ExtraTreesRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10, 25, 50, 100], 'max_depth': [3,4,5,6], 'min_samples_split': [5,10,12], 
#             'min_samples_leaf': [2,4,8]}
#EXTreeReg = ExtraTreesRegressor()
#df_temp = impute(clf=EXTreeReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=paramDict)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
paramDictAda = { 'base_estimator__max_depth':[3,4,5], 'base_estimator__min_samples_split':[5,8,12], 'base_estimator__min_samples_leaf':[2,5],
              'n_estimators': [100, 200], 'learning_rate': [.1, .5, 1] }
adaReg = AdaBoostRegressor(DecisionTreeRegressor())
df_temp = impute(clf=adaReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=paramDictAda)

#from sklearn.ensemble import RandomForestRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10,30, 50], 'max_depth': [3,4,5,6], 'min_samples_split': [5,8],
#             'min_samples_leaf': [2,4] }
#RFReg = RandomForestRegressor()
#df_temp = impute(clf=RFReg, df=df, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId', 'Survived'], paramDict=paramDict)
"""
df_noAgeNaN = df[ pd.notnull(df['Age']) ]
for index, row in df.iterrows():
  if pd.isnull(row['Age']):
    df.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )
    #print ( "\nMean=", np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] ) )
    #print ( "Medi=", np.median( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] ) )
    #print ("NewValue=", df.at[index,'Age'])

df['AgeGroup'] = df['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df['Age']
print ("AgeGroup map(Age): '-1': 0 <= x < 10 | '-.5': 10 <= x < 20 | '0': 20 <= x 30 | '.5': 30 <= x < 50 | '1': x <= 50")


#########################################
## Configuring TEST data like Train data
#########################################
df_test['Gender'] = df_test['Sex'].map( {'female': -1, 'male': 1} ).astype(int)
del df_test['Sex']

df_test['EmbarkIsC'] = df_test['Embarked'].map( lambda x: 1 if x is 'C' else -1 )
df_test['EmbarkIsS'] = df_test['Embarked'].map( lambda x: 1 if pd.isnull(x) or x is 'S' else -1 )
df_test['EmbarkIsQ'] = df_test['Embarked'].map( lambda x: 1 if x is 'Q' else -1 )
del df_test['Embarked']

df_test['TicketBeginLetter'] = df_test['Ticket'].map( lambda x: -1 if x[0].isdigit() else 1)
del df_test['Ticket']

df_test['FamilySize'] =  df_test.fillna(0)['SibSp'] + df_test.fillna(0)['Parch'] + 1
del df_test['SibSp']
del df_test['Parch']

df_test['YesMiss']   = df_test['Name'].map( lambda x: 1 if x.find('Miss.') != -1 or x.find('Ms.') != -1 or x.find('Mlle.') != -1 else -1)
df_test['YesMrs']    = df_test['Name'].map( lambda x: 1 if x.find('Mrs.') != -1  or x.find('Mme.') != -1 else -1)
df_test['YesMr']     = df_test['Name'].map( lambda x: 1 if x.find('Mr.') != -1 else -1)
df_test['YesRev']    = df_test['Name'].map( lambda x: 1 if x.find('Rev.') != -1 else -1)
df_test['YesMaster'] = df_test['Name'].map( lambda x: 1 if x.find('Master.') != -1 else -1)
df_test['YesFancy']  = df_test['Name'].map( lambda x: 1 if x.find('Countess.') != -1 or x.find('Dr.') != -1 or x.find('Capt.') != -1 or x.find('Major.') != -1
                                                                             or x.find('Don') != -1 or x.find('Col.') != -1 or x.find('Sir.') != -1 or x.find('Mme.') != -1 or x.find('Lady.') != -1
                                                                             or x.find('Jonkheer') != -1 or x.find('Dona') != -1 else -1)
del df_test['Name']                                                          

df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean() )
df_test.loc[(df_test['Fare'] <= 5), 'Fare'] = -1
df_test.loc[(df_test['Fare'] > 5) & (df_test['Fare'] <= 10), 'Fare'] = -.6
df_test.loc[(df_test['Fare'] > 10) & (df_test['Fare'] <= 25), 'Fare'] = -.2
df_test.loc[(df_test['Fare'] > 25) & (df_test['Fare'] <= 50), 'Fare'] = .2
df_test.loc[(df_test['Fare'] > 50) & (df_test['Fare'] <= 100), 'Fare'] = .6
df_test.loc[(df_test['Fare'] > 100), 'Fare'] = 1

df_test['HasCabin'] = df_test['Cabin'].map( lambda x: np.NaN if pd.isnull(x) else -1 if type(x) == float else 1)
del df_test['Cabin']


# Imputing missing values
#df_temp = impute(clf=LassReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId'], paramDict=None)
#df_temp = impute(clf=KNNReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId'], paramDict=None)
#df_temp = impute(clf=EXTreeReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId'], paramDict=None)
#df_temp = impute(clf=adaReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId'], paramDict=None) 
#df_temp = impute(clf=RFReg, df=df_test, colName='Age', isDiscrete=False, colsToRemovePrior=['PassengerId'], paramDict=None)
df_noAgeNaN = df[ pd.notnull(df['Age']) ]
for index, row in df.iterrows():
  if pd.isnull(row['Age']):
    df.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )

df_test['AgeGroup'] = df_test['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df_test['Age']

##################################
# Selecting features of interest
##################################
# All numerical or boolean features
df_noStrings = df.drop(['PassengerId'], axis=1)
df_noStrings = df_noStrings.fillna(df_noStrings.mean())

np_train_features_noStrings = df_noStrings.values
labels_train, features_train = targetFeatureSplit(np_train_features_noStrings )

np_passengerID = df_test['PassengerId'].values
df_test_noStrings = df_test.drop(['PassengerId'], axis=1)
df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mean())

features_test = df_test_noStrings.values

################
# Classifying
################
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')
from sklearn.ensemble import RandomForestClassifier
scores = defaultdict(list)

#############33 Random Forest ###################
clf = RandomForestClassifier(criterion='gini', max_depth=5, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_estimators=35)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
pred = pred.astype(int)

df_answers = pd.DataFrame({'PassengerId': np_passengerID, 'Survived': pred} )
df_answers.to_csv("Answers/titanic_answers_RF_ImpvsSimilarAge_SimilarMean.csv", sep=',', index=False)
