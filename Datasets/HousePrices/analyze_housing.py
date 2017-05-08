import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')
from formatting import *

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)


##########################
#Readinig in the csv file
##########################
df_ORI = pd.read_csv('train.csv',header=0)
#print (df_ORI.dtypes)

df = df_ORI.fillna(np.NaN )
nPoints, nColumns = df.shape

df_price = df['SalePrice']
df = df.drop(['SalePrice'], axis=1)
df_strings = df.select_dtypes(include=['object'] )
df_noStrings = df.select_dtypes(exclude=['object'])

print ( df_noStrings.shape)
print ( df_strings.shape)

print ("\n######################################\n##  Manually changing two valued features \n######################################")
df_noStrings = df_noStrings.fillna(df_noStrings.mean())
"""
df_noStrings['OverallQualAndCond'] = df_noStrings['OverallQual'] + df_noStrings['OverallCond']
df_noStrings = df_noStrings.drop(['OverallQual', 'OverallCond'], axis=1)

df_noStrings['YearsOld'] = 2017 - df_noStrings['YearRemodAdd']
df_noStrings = df_noStrings.drop(['YearBuilt', 'YearRemodAdd'], axis=1)

df_noStrings['ExterQual'] = df_strings['ExterQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_noStrings['ExterCond'] = df_strings['ExterCond'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_noStrings['ExterQualCond'] = df_noStrings['ExterQual'] + df_noStrings['ExterCond']
df_noStrings = df_noStrings.drop(['ExterQual', 'ExterCond'], axis=1)
df_strings = df_strings.drop(['ExterQual', 'ExterCond'], axis=1)

df_noStrings['BsmtQual'] = df_strings['BsmtQual'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0)
df_noStrings['BsmtCond'] = df_strings['BsmtCond'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0)
df_noStrings['BsmtExposure'] = df_strings['BsmtExposure'].map(lambda x:  np.NaN if pd.isnull(x) else 4 if x == 'Gd' else 3 if x == 'Av' else 2 if x == 'Mn' else 1 if x == 'No' else 0 )
df_noStrings['BsmtFinType1'] = df_strings['BsmtFinType1'].map(lambda x:  np.NaN if pd.isnull(x) else 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else 0 )
df_noStrings['BsmtFinType2'] = df_strings['BsmtFinType2'].map(lambda x:  np.NaN if pd.isnull(x) else 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else 0 )
df_noStrings['BasementRating'] = 3 * df_noStrings['BsmtQual'] + df_noStrings['BsmtCond'] + df_noStrings['BsmtExposure'] + 2 * df_noStrings['BsmtFinType1'] + df_noStrings['BsmtFinType1']
df_noStrings = df_noStrings.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis=1)
df_strings = df_strings.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis=1)

df_noStrings['HeatingQC'] = df_strings['HeatingQC'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_strings = df_strings.drop(['HeatingQC'], axis=1)

df_noStrings['KitchenQual'] = df_strings['KitchenQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_strings = df_strings.drop(['KitchenQual'], axis=1)

df_noStrings['FireplaceQu'] = df_strings['FireplaceQu'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_noStrings['nFireplacesAndQuality'] = df_noStrings['FireplaceQu'] * df_noStrings['Fireplaces']
df_noStrings = df_noStrings.drop(['FireplaceQu','Fireplaces'], axis=1)
df_strings = df_strings.drop(['FireplaceQu'], axis=1)

df_noStrings['GarageFinish'] = df_strings['GarageFinish'].map( lambda x: np.NaN if x == pd.isnull(x) else 3 if x == 'Fin' else 2 if x == 'RFn' else 1 if x == 'Unf' else 0 )
df_noStrings['GarageQual'] = df_strings['GarageQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_noStrings['GarageCond'] = df_strings['GarageCond'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_noStrings['PavedDrive'] = df_strings['PavedDrive'].map( lambda x:np.NaN if pd.isnull(x) else 2 if x == 'Y' else 1 if x == 'P' else 0 )
df_noStrings['GarageRating'] = df_noStrings['GarageFinish'] + df_noStrings['GarageQual'] + df_noStrings['GarageCond'] + df_noStrings['PavedDrive'] 
df_noStrings = df_noStrings.drop(['GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'] , axis=1)
df_strings = df_strings.drop(['GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'], axis=1)

df_noStrings['PoolQC'] = df_strings['PoolQC'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 0 )
df_noStrings['Fence'] = df_strings['Fence'].map( lambda x: np.NaN if pd.isnull(x) else 4 if x == 'GdPrv' else 3 if x == 'MnPrv' else 2 if x == 'GdWo' else 1 if x == 'MnWw' else 0 )
df_strings = df_strings.drop(['PoolQC', 'Fence'], axis=1)

df_noStrings['LotShape'] = df_strings['LotShape'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x == 'Reg' else -1 if x == 'IR1' else -2 if x == 'IR2' else -3)
df_strings = df_strings.drop(['LotShape'], axis=1)

df_noStrings['Utilities'] = df_strings['Utilities'].map( lambda x: np.NaN if pd.isnull(x) else 2 if x == 'AllPub' else 1 if x == 'NoSewr' else -1 if x == 'NoSeWa' else -2 )
df_strings = df_strings.drop(['Utilities'], axis=1)

df_noStrings['LandSlope'] = df_strings['LandSlope'].map( lambda x: np.NaN if pd.isnull(x) else 3 if x == 'Gtl' else 2 if x == 'Mod' else 0)
df_strings = df_strings.drop(['LandSlope'], axis=1)

df_noStrings['Bathrooms'] = df_noStrings['BsmtFullBath'] + .33*df_noStrings['BsmtHalfBath'] + 2*(df_noStrings['FullBath'] + .33*df_noStrings['HalfBath'])
df_noStrings = df_noStrings.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)

df_noStrings['PorchArea'] = df_noStrings['EnclosedPorch'] + df_noStrings['3SsnPorch'] + df_noStrings['ScreenPorch'] + df_noStrings['OpenPorchSF'] + df_noStrings['WoodDeckSF']
df_noStrings = df_noStrings.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF'], axis=1)

df_noStrings = df_noStrings.drop(['MoSold'], axis=1)


df_noStrings = df_noStrings.fillna(df_noStrings.mode())
print ( df_noStrings.shape)
print ( df_strings.shape)
"""

print ("\n######################################\n##  Automatically changing two valued features \n######################################")
for column in df_strings:
  nUnique = df_strings[column].unique()
  print ("column=", column, "\tlen(column)=", len(df_strings[column]), "\tunique=", nUnique)
  if len(nUnique) == 2:
    print ("\t", nUnique[0], "=1\t", nUnique[1],"=-1")
    df_noStrings = pd.concat([df_noStrings, df_strings[column].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is nUnique[0] else -1 ) ], axis=1)
    del df_strings[column]
  elif len(nUnique) <= 4:
    for i in nUnique:
      if not pd.isnull(i):
        newColumnName = column + "Yes_" + i
        df_noStrings[newColumnName] = df_strings[column].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is i else -1 ) 
        print ("\t", newColumnName, "\tMode=", df_noStrings[newColumnName].mode()[0] )
        df_noStrings[newColumnName].fillna(df_noStrings[newColumnName].mode()[0] )
    del df_strings[column]
  else:
    nValues=1
    uniqueDict = {}
    for i in nUnique:
      if pd.isnull(i):
        uniqueDict[i] = np.NaN
      else:
        uniqueDict[i] = nValues 
        nValues += 1
        print ("\t", i, "=", nValues-1)
    df_noStrings[column] = df_strings[column].map(uniqueDict)
    print ("\tMode=", df_noStrings[column].mode()[0] )
    df_noStrings[column].fillna(df_noStrings[column].mode()[0] )
    del df_strings[column]

df_noStrings = df_noStrings.fillna(df_noStrings.mean())
df_noStrings_means = df_noStrings.mean()
print ("\n\n###################### DF MEANS ###########3\n", df_noStrings_means)
df_noStrings = df_noStrings - df_noStrings.mean()
df_noStrings_std = df_noStrings.std()
print ("\n\n###################### DF STD  #############\n",  df_noStrings_std )
df_noStrings = df_noStrings / df_noStrings.std()

print (df_noStrings.describe() )

##############
## Analyzing
##############
#df_important_features = df_noStrings[['LotArea', 'TotalBsmtSF', 'BasementRating', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BedroomAbvGr', 'GarageArea', 'GarageCars', 'PorchArea', 'Bathrooms',  'GarageRating', 'MSZoning', 'AlleyYes_Grvl', 'AlleyYes_Pave', 'LandContourYes_Lvl', 'LandContourYes_Bnk', 'LandContourYes_Low', 'LandContourYes_HLS', 'Condition1', 'Condition2', 'BldgType', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Foundation', 'CentralAir', 'HeatingQC', 'nFireplacesAndQuality', 'YearsOld', 'OverallQualAndCond', 'KitchenQual', 'PoolQC', 'Fence', 'LotShape', 'Utilities', 'LandSlope', 'GarageType', 'Functional', 'SaleType', 'SaleCondition']]

df_all = pd.concat([df_price, df_noStrings], axis =1)
np_all = df_all.values
labels_all, features_all = targetFeatureSplit(np_all) 

#df_selection = pd.concat( [df_price, df_important_features], axis=1)
#np_selection = df_selection.values
#labels_selection, features_selection = targetFeatureSplit(np_selection )

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features_all, labels_all, test_size=0.25, random_state=42)

from sklearn import linear_model
from time import time
from GridSearch import *

############ Ordinary Linear Regression ############
lineReg = linear_model.LinearRegression()
lineReg.fit(features_train, labels_train)
pred = lineReg.predict(features_test)
acc = lineReg.score(features_test, labels_test)
print ("\n\nLinear Regression: accuracy=", acc, "\ncoef_=", lineReg.coef_ )


############ Ridge Linear Regression ############
#paramDict = {"alpha": [.1, .5, 1, 1.5, 2, 10, 20 ], 'tol': [.0000001, .00001, .0001, .001, .01]}
ridgeReg = linear_model.Ridge(tol=1e-07, alpha= 20)
ridgeReg.fit(features_train, labels_train)
pred = ridgeReg.predict(features_test)
acc = ridgeReg.score(features_test, labels_test)
print ("\n\nRidge Linear Regression: accuracy=", acc, "\ncoef_=", ridgeReg.coef_ )


############ Elastic Net Regression ############
#paramDict = {'alpha': [.1, 1, 2, 5, 10], 'l1_ratio': [.1, .25, .5, .75, .9], 'tol': [.0000001, .00001, .0001, .001, .01] }
elasticReg = linear_model.ElasticNet(tol=0.01, alpha=0.1, l1_ratio=0.25)
elasticReg.fit(features_train, labels_train)
pred = elasticReg.predict(features_test)
acc = elasticReg.score(features_test, labels_test)
print ("\n\nElastic Net Regression: accuracy=", acc, "\ncoef_=", elasticReg.coef_ )


###########  Baysean  ARD Regression ###########
#paramDict = {'tol': [ .0001, .001], 'alpha_1': [.000001, .00001], 'alpha_2': [.000001, .00001], 'lambda_1': [.000001, .00001], 'lambda_2': [.000001, .00001, ], 'threshold_lambda': [ 1000, 10000]}
ARDReg = linear_model.ARDRegression(alpha_1=.000001, alpha_2=.000001, lambda_1=.000001, lambda_2=.000001, threshold_lambda=1000, tol=.0001)
ARDReg.fit(features_train, labels_train)
pred = ARDReg.predict(features_test)
acc = ARDReg.score(features_test, labels_test)
print ("\n\nARD Regression 1: accuracy=", acc, "\ncoef_=", ARDReg.coef_ )

ARDReg = linear_model.ARDRegression(alpha_1=.00001, alpha_2=.00001, lambda_1=.00001, lambda_2=.000001, threshold_lambda=10000, tol=.00001)
ARDReg.fit(features_train, labels_train)
pred = ARDReg.predict(features_test)
acc = ARDReg.score(features_test, labels_test)
print ("\n\nARD Regression 2: accuracy=", acc, "\ncoef_=", ARDReg.coef_ )

ARDReg = linear_model.ARDRegression(alpha_1=.000001, alpha_2=.000001, lambda_1=.000001, lambda_2=.000001, threshold_lambda=10000, tol=.00001)
ARDReg.fit(features_train, labels_train)
pred = ARDReg.predict(features_test)
acc = ARDReg.score(features_test, labels_test)
print ("\n\nARD Regression 3: accuracy=", acc, "\ncoef_=", ARDReg.coef_ )

ARDReg = linear_model.ARDRegression(alpha_1=.00001, alpha_2=.00001, lambda_1=.00001, lambda_2=.000001, threshold_lambda=1000, tol=.0001)
ARDReg.fit(features_train, labels_train)
pred = ARDReg.predict(features_test)
acc = ARDReg.score(features_test, labels_test)
print ("\n\nARD Regression 4: accuracy=", acc, "\ncoef_=", ARDReg.coef_ )


############ Stochastic Gradient Descent Regressor ############
#paramDict = {'loss': ['squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'penalty': ['l2', 'elasticnet', None], 'alpha': [.00001, .0001 ], 'epsilon': [.1, .05, .01], 'learning_rate': ['optimal', 'invscaling'] }
SGDReg = linear_model.SGDRegressor(alpha=1e-05, penalty='l2', loss='squared_loss', learning_rate='invscaling', epsilon=0.01)
SGDReg.fit(features_train, labels_train)
pred = SGDReg.predict(features_test)
acc = SGDReg.score(features_test, labels_test)
print ("\n\nSGD Regression: accuracy=", acc, "\ncoef_=", SGDReg.coef_ )
####### BestScore= 0.787316849183 with these best Parameters= {'alpha': 1e-05, 'penalty': 'l2', 'loss': 'squared_loss', 'learning_rate': 'invscaling', 'epsilon': 0.01}


############ Passive-Aggressive Regressor  ############
#paramDict = {'C': [.1, 1, 10, 25, 50], 'epsilon': [.02, .05, .01, .001], 'n_iter': [3, 5, 10, 20], 'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'] }
passReg =  linear_model.PassiveAggressiveRegressor(C=25, loss='epsilon_insensitive', n_iter=10, epsilon=0.02)
passReg.fit(features_train, labels_train)
pred = passReg.predict(features_test)
acc = passReg.score(features_test, labels_test)
print ("\n\npass Regression 3: accuracy=", acc, "\ncoef_=", passReg.coef_ )
########   BestScore= 0.79440546561  with these best Parameters= {'C': 50, 'loss': 'epsilon_insensitive', 'n_iter': 10, 'epsilon': 0.02}


############ Polynomial  ############
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
pipe2 = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
pipe3 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
pipe2.fit(features_train, labels_train)
pred = pipe2.predict(features_test)
acc = pipe2.score(features_test, labels_test)
print ("\n\nPoly2 Regression: accuracy=", acc)#, "\ncoef_ type=", pipe2.named_steps['linear'].coef_ )

pipe3.fit(features_train, labels_train)
pred = pipe3.predict(features_test)
acc = pipe3.score(features_test, labels_test)
print ("\n\npoly3 Regression: accuracy=", acc)#, "\nfeature_importances_=", pipe3.get_params )

############ DecisionTree Regressor ############
from sklearn.tree import DecisionTreeRegressor
decTreeReg = DecisionTreeRegressor(min_samples_split=5, max_depth=12, min_samples_leaf=3, min_impurity_split=1e-06)
decTreeReg.fit(features_train, labels_train)
pred = decTreeReg.predict(features_test)
acc = decTreeReg.score(features_test, labels_test)
print ("\n\nDecision Tree Regression: accuracy=", acc, "\nfeature_importances_=", decTreeReg.feature_importances_ )
###### BestScore= 0.725369650414 with these best Parameters= {'min_samples_split': 5, 'min_samples_leaf': 3, 'min_impurity_split': 1e-06, 'max_depth': 12}


############ Ada Boost ############
from sklearn.ensemble import AdaBoostRegressor
paramDict = { 'base_estimator__max_depth':[4, 8], 'base_estimator__min_samples_split':[4,12], 'base_estimator__min_samples_leaf':[2,5], 
              'base_estimator__min_impurity_split': [.0000001, .000001], 'n_estimators': [100, 200], 'learning_rate': [.1, 1] }
ada1Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=8, min_samples_leaf=5), n_estimators=150, learning_rate=.1 )
ada1Reg.fit(features_train, labels_train)
pred = ada1Reg.predict(features_test)
acc = ada1Reg.score(features_test, labels_test)
print ("\n\nada1 Regression: accuracy=", acc, "\nfeature_importances_=", ada1Reg.feature_importances_ )

ada2Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=8, min_samples_leaf=5), n_estimators=300, learning_rate=.1 )
ada2Reg.fit(features_train, labels_train)
pred = ada2Reg.predict(features_test)
acc = ada2Reg.score(features_test, labels_test)
print ("\n\nada2 Regression: accuracy=", acc, "\nfeature_importances_=", ada2Reg.feature_importances_ )

ada3Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=8, min_samples_leaf=5), n_estimators=150, learning_rate=1 )
ada3Reg.fit(features_train, labels_train)
pred = ada3Reg.predict(features_test)
acc = ada3Reg.score(features_test, labels_test)
print ("\n\nada3 Regression: accuracy=", acc, "\nfeature_importances_=", ada3Reg.feature_importances_ )

ada4Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=8, min_samples_leaf=5), n_estimators=300, learning_rate=1 )
ada4Reg.fit(features_train, labels_train)
pred = ada4Reg.predict(features_test)
acc = ada4Reg.score(features_test, labels_test)
print ("\n\nada4 Regression: accuracy=", acc, "\nfeature_importances_=", ada4Reg.feature_importances_ )

ada5Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_split=8, min_samples_leaf=3), n_estimators=150, learning_rate=.1 )
ada5Reg.fit(features_train, labels_train)
pred = ada5Reg.predict(features_test)
acc = ada5Reg.score(features_test, labels_test)
print ("\n\nada5 Regression: accuracy=", acc, "\nfeature_importances_=", ada5Reg.feature_importances_ )


ada6Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_split=8, min_samples_leaf=3), n_estimators=300, learning_rate=.1 )
ada6Reg.fit(features_train, labels_train)
pred = ada6Reg.predict(features_test)
acc = ada6Reg.score(features_test, labels_test)
print ("\n\nada6 Regression: accuracy=", acc, "\nfeature_importances_=", ada6Reg.feature_importances_ )

ada7Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_split=8, min_samples_leaf=3), n_estimators=150, learning_rate=1 )
ada7Reg.fit(features_train, labels_train)
pred = ada7Reg.predict(features_test)
acc = ada7Reg.score(features_test, labels_test)
print ("\n\nada7 Regression: accuracy=", acc, "\nfeature_importances_=", ada7Reg.feature_importances_ )

ada8Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_split=8, min_samples_leaf=3), n_estimators=300, learning_rate=1 )
ada8Reg.fit(features_train, labels_train)
pred = ada8Reg.predict(features_test)
acc = ada8Reg.score(features_test, labels_test)
print ("\n\nada8 Regression: accuracy=", acc, "\nfeature_importances_=", ada8Reg.feature_importances_ )


############ ExtraTreesRegressor ############
from sklearn.ensemble import ExtraTreesRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10, 25, 50], 'max_depth': [4,8,12,None], 'min_samples_split': [2,5,8], 
#             'min_samples_leaf': [2,3,4], 'min_impurity_split': [.0000001, .000001]}
extraTrees1Reg = ExtraTreesRegressor(criterion='mse', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
extraTrees1Reg.fit(features_train, labels_train)
pred = extraTrees1Reg.predict(features_test)
acc = extraTrees1Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 1 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees1Reg.feature_importances_ )

extraTrees2Reg = ExtraTreesRegressor(criterion='mse', n_estimators=30, max_depth=7, min_samples_split=4, min_samples_leaf=2)
extraTrees2Reg.fit(features_train, labels_train)
pred = extraTrees2Reg.predict(features_test)
acc = extraTrees2Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 2 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees2Reg.feature_importances_ )

extraTrees3Reg = ExtraTreesRegressor(criterion='mae', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
extraTrees3Reg.fit(features_train, labels_train)
pred = extraTrees3Reg.predict(features_test)
acc = extraTrees3Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 3 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees3Reg.feature_importances_ )

extraTrees4Reg = ExtraTreesRegressor(criterion='mae', n_estimators=30, max_depth=7, min_samples_split=4, min_samples_leaf=2)
extraTrees4Reg.fit(features_train, labels_train)
pred = extraTrees4Reg.predict(features_test)
acc = extraTrees4Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 4 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees4Reg.feature_importances_ )

extraTrees5Reg = ExtraTreesRegressor(criterion='mse', n_estimators=15, max_depth=4, min_samples_split=9, min_samples_leaf=4)
extraTrees5Reg.fit(features_train, labels_train)
pred = extraTrees5Reg.predict(features_test)
acc = extraTrees5Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 5 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees5Reg.feature_importances_ )

extraTrees6Reg = ExtraTreesRegressor(criterion='mse', n_estimators=30, max_depth=4, min_samples_split=9, min_samples_leaf=4)
extraTrees6Reg.fit(features_train, labels_train)
pred = extraTrees6Reg.predict(features_test)
acc = extraTrees6Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 6 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees6Reg.feature_importances_ )

extraTrees7Reg = ExtraTreesRegressor(criterion='mae', n_estimators=15, max_depth=4, min_samples_split=9, min_samples_leaf=4)
extraTrees7Reg.fit(features_train, labels_train)
pred = extraTrees7Reg.predict(features_test)
acc = extraTrees7Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 7 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees7Reg.feature_importances_ )

extraTrees8Reg = ExtraTreesRegressor(criterion='mae', n_estimators=30, max_depth=4, min_samples_split=9, min_samples_leaf=4)
extraTrees8Reg.fit(features_train, labels_train)
pred = extraTrees8Reg.predict(features_test)
acc = extraTrees8Reg.score(features_test, labels_test)
print ("\n\nExtra Tree 8 Regression: accuracy=", acc, "\nfeature_importances_=", extraTrees8Reg.feature_importances_ )


############ Random forest Regressor ############
from sklearn.ensemble import RandomForestRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10,0, 50], 'max_depth': [4,8,12,None], 'min_samples_split': [2,5,8],
#             'min_samples_leaf': [2,3,4], 'min_impurity_split': [.0000001, .000001] }
randomTrees1Reg = RandomForestRegressor(criterion='mse', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
randomTrees1Reg.fit(features_train, labels_train)
pred = randomTrees1Reg.predict(features_test)
acc = randomTrees1Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 1 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees1Reg.feature_importances_ )

randomTrees2Reg = RandomForestRegressor(criterion='mse', n_estimators=30, max_depth=7, min_samples_split=4, min_samples_leaf=2)
randomTrees2Reg.fit(features_train, labels_train)
pred = randomTrees2Reg.predict(features_test)
acc = randomTrees2Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 2 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees2Reg.feature_importances_ )

randomTrees3Reg = RandomForestRegressor(criterion='mae', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
randomTrees3Reg.fit(features_train, labels_train)
pred = randomTrees3Reg.predict(features_test)
acc = randomTrees3Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 3 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees3Reg.feature_importances_ )

randomTrees4Reg = RandomForestRegressor(criterion='mae', n_estimators=30, max_depth=7, min_samples_split=4, min_samples_leaf=2)
randomTrees4Reg.fit(features_train, labels_train)
pred = randomTrees4Reg.predict(features_test)
acc = randomTrees4Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 4 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees4Reg.feature_importances_ )

randomTrees5Reg = RandomForestRegressor(criterion='mse', n_estimators=15, max_depth=4, min_samples_split=9, min_samples_leaf=4)
randomTrees5Reg.fit(features_train, labels_train)
pred = randomTrees5Reg.predict(features_test)
acc = randomTrees5Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 5 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees5Reg.feature_importances_ )

randomTrees6Reg = RandomForestRegressor(criterion='mse', n_estimators=30, max_depth=4, min_samples_split=9, min_samples_leaf=4)
randomTrees6Reg.fit(features_train, labels_train)
pred = randomTrees6Reg.predict(features_test)
acc = randomTrees6Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 6 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees6Reg.feature_importances_ )

randomTrees7Reg = RandomForestRegressor(criterion='mae', n_estimators=15, max_depth=4, min_samples_split=9, min_samples_leaf=4)
randomTrees7Reg.fit(features_train, labels_train)
pred = randomTrees7Reg.predict(features_test)
acc = randomTrees7Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 7 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees7Reg.feature_importances_ )

randomTrees8Reg = RandomForestRegressor(criterion='mae', n_estimators=30, max_depth=4, min_samples_split=9, min_samples_leaf=4)
randomTrees8Reg.fit(features_train, labels_train)
pred = randomTrees8Reg.predict(features_test)
acc = randomTrees8Reg.score(features_test, labels_test)
print ("\n\nRandom Tree 8 Regression: accuracy=", acc, "\nfeature_importances_=", randomTrees8Reg.feature_importances_ )

