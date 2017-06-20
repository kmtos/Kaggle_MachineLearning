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
df = df_ORI.copy()
df = df.fillna(np.NaN )

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

df_noAgeNaN = df[ pd.notnull(df['Age']) ]
for index, row in df.iterrows():
  if pd.isnull(row['Age']):
    df.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )
df['AgeGroup'] = df['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df['Age']

df_noAgeNaN = df_test[ pd.notnull(df_test['Age']) ]
for index, row in df_test.iterrows():
  if pd.isnull(row['Age']):
    df_test.at[index,'Age'] = np.mean( df_noAgeNaN.loc[(df_noAgeNaN['Pclass'] == row['Pclass']) & (df_noAgeNaN['Gender'] == row['Gender']) & (pd.notnull(df_noAgeNaN['Age'] )), "Age"] )
df_test['AgeGroup'] = df_test['Age'].map(lambda x: -1 if x >= 0 and x < 10 else -.5 if x >= 10 and x < 20 else 0 if x >= 20 and x < 30 else .5 if x >= 30 and x < 50 else 1)
del df_test['Age']


##################################
# Selecting features of interest
##################################
# All numerical or boolean features
df_noStrings = df.drop([], axis=1)
df_noStrings = df_noStrings.fillna(df_noStrings.mean())

np_train_features_noStrings = df_noStrings.values
labels_train, features_train = targetFeatureSplit(np_train_features_noStrings )

from sklearn.model_selection import train_test_split
features_train_1, features_test_1, labels_train_1, labels_test_1 = train_test_split(features_train, labels_train, test_size=0.25, random_state=42)

np_passengerID = df_test['PassengerId'].values
df_test_noStrings = df_test.drop(['PassengerId'], axis=1)
df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mean())

features_test = df_test_noStrings.values

#########################
# Testing my Fuzzy Trees
#########################
from FuzzyTrees import *
MakeTree(df=df_noStrings, className='Survived', nGiniSplits=10, alpha=.1, giniEndVal=.001, maxDepth=4, idColumn='PassengerId', outputDir='/home/kyletos/Kaggle/Algorithms/TestTheFuzzy/Answers/')



 
