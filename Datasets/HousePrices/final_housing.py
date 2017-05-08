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

df_noStrings = df_noStrings.fillna(df_noStrings.mean())

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


#################
## Testing Data
#################
df_test_ORI = pd.read_csv('test.csv',header=0)
#print (df_test_ORI.dtypes)

df_test = df_test_ORI.fillna(np.NaN )
nPoints, nColumns = df_test.shape

df_test_strings = df_test.select_dtypes(include=['object'] )
df_test_noStrings = df_test.select_dtypes(exclude=['object'])

print ( df_test_noStrings.shape)
print ( df_test_strings.shape)

df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mean())

df_test_noStrings['OverallQualAndCond'] = df_test_noStrings['OverallQual'] + df_test_noStrings['OverallCond']
df_test_noStrings = df_test_noStrings.drop(['OverallQual', 'OverallCond'], axis=1)

df_test_noStrings['YearsOld'] = 2017 - df_test_noStrings['YearRemodAdd']
df_test_noStrings = df_test_noStrings.drop(['YearBuilt', 'YearRemodAdd'], axis=1)

df_test_noStrings['ExterQual'] = df_test_strings['ExterQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_test_noStrings['ExterCond'] = df_test_strings['ExterCond'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_test_noStrings['ExterQualCond'] = df_test_noStrings['ExterQual'] + df_test_noStrings['ExterCond']
df_test_noStrings = df_test_noStrings.drop(['ExterQual', 'ExterCond'], axis=1)
df_test_strings = df_test_strings.drop(['ExterQual', 'ExterCond'], axis=1)

df_test_noStrings['BsmtQual'] = df_test_strings['BsmtQual'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0)
df_test_noStrings['BsmtCond'] = df_test_strings['BsmtCond'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0)
df_test_noStrings['BsmtExposure'] = df_test_strings['BsmtExposure'].map(lambda x:  np.NaN if pd.isnull(x) else 4 if x == 'Gd' else 3 if x == 'Av' else 2 if x == 'Mn' else 1 if x == 'No' else 0 )
df_test_noStrings['BsmtFinType1'] = df_test_strings['BsmtFinType1'].map(lambda x:  np.NaN if pd.isnull(x) else 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else 0 )
df_test_noStrings['BsmtFinType2'] = df_test_strings['BsmtFinType2'].map(lambda x:  np.NaN if pd.isnull(x) else 6 if x == 'GLQ' else 5 if x == 'ALQ' else 4 if x == 'BLQ' else 3 if x == 'Rec' else 2 if x == 'LwQ' else 1 if x == 'Unf' else 0 )
df_test_noStrings['BasementRating'] = 3 * df_test_noStrings['BsmtQual'] + df_test_noStrings['BsmtCond'] + df_test_noStrings['BsmtExposure'] + 2 * df_test_noStrings['BsmtFinType1'] + df_test_noStrings['BsmtFinType1']
df_test_noStrings = df_test_noStrings.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis=1)
df_test_strings = df_test_strings.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis=1)

df_test_noStrings['HeatingQC'] = df_test_strings['HeatingQC'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_test_strings = df_test_strings.drop(['HeatingQC'], axis=1)

df_test_noStrings['KitchenQual'] = df_test_strings['KitchenQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 )
df_test_strings = df_test_strings.drop(['KitchenQual'], axis=1)

df_test_noStrings['FireplaceQu'] = df_test_strings['FireplaceQu'].map(lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_test_noStrings['nFireplacesAndQuality'] = df_test_noStrings['FireplaceQu'] * df_test_noStrings['Fireplaces']
df_test_noStrings = df_test_noStrings.drop(['FireplaceQu','Fireplaces'], axis=1)
df_test_strings = df_test_strings.drop(['FireplaceQu'], axis=1)

df_test_noStrings['GarageFinish'] = df_test_strings['GarageFinish'].map( lambda x: np.NaN if x == pd.isnull(x) else 3 if x == 'Fin' else 2 if x == 'RFn' else 1 if x == 'Unf' else 0 )
df_test_noStrings['GarageQual'] = df_test_strings['GarageQual'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_test_noStrings['GarageCond'] = df_test_strings['GarageCond'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 1 if x == 'Po' else 0 )
df_test_noStrings['PavedDrive'] = df_test_strings['PavedDrive'].map( lambda x:np.NaN if pd.isnull(x) else 2 if x == 'Y' else 1 if x == 'P' else 0 )
df_test_noStrings['GarageRating'] = df_test_noStrings['GarageFinish'] + df_test_noStrings['GarageQual'] + df_test_noStrings['GarageCond'] + df_test_noStrings['PavedDrive'] 
df_test_noStrings = df_test_noStrings.drop(['GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'] , axis=1)
df_test_strings = df_test_strings.drop(['GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'], axis=1)

df_test_noStrings['PoolQC'] = df_test_strings['PoolQC'].map( lambda x: np.NaN if pd.isnull(x) else 5 if x == 'Ex' else 4 if x == 'Gd' else 3 if x == 'TA' else 2 if x == 'Fa' else 0 )
df_test_noStrings['Fence'] = df_test_strings['Fence'].map( lambda x: np.NaN if pd.isnull(x) else 4 if x == 'GdPrv' else 3 if x == 'MnPrv' else 2 if x == 'GdWo' else 1 if x == 'MnWw' else 0 )
df_test_strings = df_test_strings.drop(['PoolQC', 'Fence'], axis=1)

df_test_noStrings['LotShape'] = df_test_strings['LotShape'].map( lambda x: np.NaN if pd.isnull(x) else 1 if x == 'Reg' else -1 if x == 'IR1' else -2 if x == 'IR2' else -3)
df_test_strings = df_test_strings.drop(['LotShape'], axis=1)

df_test_noStrings['Utilities'] = df_test_strings['Utilities'].map( lambda x: np.NaN if pd.isnull(x) else 2 if x == 'AllPub' else 1 if x == 'NoSewr' else -1 if x == 'NoSeWa' else -2 )
df_test_strings = df_test_strings.drop(['Utilities'], axis=1)

df_test_noStrings['LandSlope'] = df_test_strings['LandSlope'].map( lambda x: np.NaN if pd.isnull(x) else 3 if x == 'Gtl' else 2 if x == 'Mod' else 0)
df_test_strings = df_test_strings.drop(['LandSlope'], axis=1)

df_test_noStrings['Bathrooms'] = df_test_noStrings['BsmtFullBath'] + .33*df_test_noStrings['BsmtHalfBath'] + 2*(df_test_noStrings['FullBath'] + .33*df_test_noStrings['HalfBath'])
df_test_noStrings = df_test_noStrings.drop(['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'], axis=1)

df_test_noStrings['PorchArea'] = df_test_noStrings['EnclosedPorch'] + df_test_noStrings['3SsnPorch'] + df_test_noStrings['ScreenPorch'] + df_test_noStrings['OpenPorchSF'] + df_test_noStrings['WoodDeckSF']
df_test_noStrings = df_test_noStrings.drop(['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'OpenPorchSF', 'WoodDeckSF'], axis=1)

df_test_noStrings = df_test_noStrings.drop(['MoSold'], axis=1)

df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mode())

print ( df_test_noStrings.shape)
print ( df_test_strings.shape)


################################
# Automatically changing values
################################
print ("\n######################################\n##  Automatically changing two valued features \n######################################")
for column in df_strings:
  nUnique = df_strings[column].unique()
  print ("column=", column, "\tlen(column)=", len(df_strings[column]), "\tunique=", nUnique)
  if len(nUnique) == 2:
    print ("\t", nUnique[0], "=1\t", nUnique[1],"=-1")
    df_noStrings      = pd.concat([df_noStrings,     df_strings[column].map(       lambda x: np.NaN if pd.isnull(x) else 1 if x is nUnique[0] else -1 ) ], axis=1)
    df_test_noStrings = pd.concat([df_test_noStrings, df_test_strings[column].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is nUnique[0] else -1 ) ], axis=1)
    del df_strings[column]
    del df_test_strings[column]
  elif len(nUnique) <= 4:
    for i in nUnique:
      if not pd.isnull(i):
        newColumnName = column + "Yes_" + i
        df_noStrings[newColumnName] = df_strings[column].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is i else -1 ) 
        print ("\t", newColumnName, "\tMode=", df_noStrings[newColumnName].mode()[0] )
        df_noStrings[newColumnName].fillna(df_noStrings[newColumnName].mode()[0] )

        df_test_noStrings[newColumnName] = df_test_strings[column].map( lambda x: np.NaN if pd.isnull(x) else 1 if x is i else -1 )
        print ("\t", newColumnName, "\tMode=", df_test_noStrings[newColumnName].mode()[0] )
        df_test_noStrings[newColumnName].fillna(df_test_noStrings[newColumnName].mode()[0] )

    del df_strings[column]
    del df_test_strings[column]
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

    df_test_noStrings[column] = df_test_strings[column].map(uniqueDict)
    df_test_noStrings[column].fillna(df_test_noStrings[column].mode()[0] )
    del df_test_strings[column]

print ("\n\n", df_noStrings.describe() , "\n\n")

df_noStrings = df_noStrings.fillna(df_noStrings.mean())
df_noStrings_means = df_noStrings.mean()
df_noStrings = df_noStrings - df_noStrings.mean()
df_noStrings_std = df_noStrings.std()
df_noStrings = df_noStrings / df_noStrings.std()

print (df_test_noStrings.describe() )

df_test_noStrings = df_test_noStrings.fillna(df_test_noStrings.mean())
df_test_noStrings_means = df_test_noStrings.mean()
for column in df_test_noStrings:
  if df_test_noStrings[column].std() != 0:
    df_test_noStrings[column] = df_test_noStrings[column] - df_test_noStrings[column].mean()
    df_test_noStrings[column] = df_test_noStrings[column] / df_test_noStrings[column].std()
print ("\n\n\n\n\n\n", df_test_noStrings.describe() )
###################
## Analyzing Train
###################
df_important_features = df_noStrings[['LotArea', 'TotalBsmtSF', 'BasementRating', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BedroomAbvGr', 'GarageArea', 'GarageCars', 'PorchArea', 'Bathrooms',  'GarageRating', 'MSZoning', 'AlleyYes_Grvl', 'AlleyYes_Pave', 'LandContourYes_Lvl', 'LandContourYes_Bnk', 'LandContourYes_Low', 'LandContourYes_HLS', 'Condition1', 'Condition2', 'BldgType', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Foundation', 'CentralAir', 'HeatingQC', 'nFireplacesAndQuality', 'YearsOld', 'OverallQualAndCond', 'KitchenQual', 'PoolQC', 'Fence', 'LotShape', 'Utilities', 'LandSlope', 'GarageType', 'Functional', 'SaleType', 'SaleCondition']]

df_all = pd.concat([df_price, df_noStrings], axis =1)
np_all = df_all.values
labels_all, features_all = targetFeatureSplit(np_all) 

df_selection = pd.concat( [df_price, df_important_features], axis=1)
np_selection = df_selection.values
labels_selection, features_selection = targetFeatureSplit(np_selection )

##################
## Analyzing TEst
##################
df_test_important_features = df_test_noStrings[['LotArea', 'TotalBsmtSF', 'BasementRating', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BedroomAbvGr', 'GarageArea', 'GarageCars', 'PorchArea', 'Bathrooms',  'GarageRating', 'MSZoning', 'AlleyYes_Grvl', 'AlleyYes_Pave', 'LandContourYes_Lvl', 'LandContourYes_Bnk', 'LandContourYes_Low', 'LandContourYes_HLS', 'Condition1', 'Condition2', 'BldgType', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 'Foundation', 'CentralAir', 'HeatingQC', 'nFireplacesAndQuality', 'YearsOld', 'OverallQualAndCond', 'KitchenQual', 'PoolQC', 'Fence', 'LotShape', 'Utilities', 'LandSlope', 'GarageType', 'Functional', 'SaleType', 'SaleCondition']]

np_test_all = df_test_noStrings.values
np_test_selection = df_test_important_features.values

#################
# Learndf_test_important_featuresing
################
from sklearn import linear_model
from time import time
from GridSearch import *

"""
############ Stochastic Gradient Descent Regressor ############
#paramDict = {'loss': ['squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'penalty': ['l2', 'elasticnet', None], 'alpha': [.00001, .0001 ], 'epsilon': [.1, .05, .01], 'learning_rate': ['optimal', 'invscaling'] }
SGDReg = linear_model.SGDRegressor(alpha=1e-05, penalty='l2', loss='squared_loss', learning_rate='invscaling', epsilon=0.01)
SGDReg.fit(features_selection, labels_selection)
pred = SGDReg.predict(features_test)
acc = SGDReg.score(features_test, labels_test)
print ("\n\nSGD Regression: accuracy=", acc, "\ncoef_=", SGDReg.coef_ )
####### BestScore= 0.787316849183 with these best Parameters= {'alpha': 1e-05, 'penalty': 'l2', 'loss': 'squared_loss', 'learning_rate': 'invscaling', 'epsilon': 0.01}
"""

############ Ada Boost ############
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
#paramDict = { 'base_estimator__max_depth':[4, 8], 'base_estimator__min_samples_split':[4,12], 'base_estimator__min_samples_leaf':[2,5], 
#              'base_estimator__min_impurity_split': [.0000001, .000001], 'n_estimators': [100, 200], 'learning_rate': [.1, 1] }

ada3Reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8, min_samples_split=8, min_samples_leaf=5), n_estimators=150, learning_rate=1 )
ada3Reg.fit(features_all, labels_all)
pred = ada3Reg.predict(np_test_all)
"""
############ ExtraTreesRegressor ############
from sklearn.ensemble import ExtraTreesRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10, 25, 50], 'max_depth': [4,8,12,None], 'min_samples_split': [2,5,8], 
#             'min_samples_leaf': [2,3,4], 'min_impurity_split': [.0000001, .000001]}
extraTrees3Reg = ExtraTreesRegressor(criterion='mae', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
extraTrees3Reg.fit(features_selection, labels_selection)
pred = extraTrees3Reg.predict(np_test_selection)

############ Random forest Regressor ############
from sklearn.ensemble import RandomForestRegressor
#paramDict = {'criterion': ['mse', 'mae'], 'n_estimators': [10,0, 50], 'max_depth': [4,8,12,None], 'min_samples_split': [2,5,8],samples_leaf': [2,3,4], 'min_impurity_split': [.0000001, .000001] }

randomTrees3Reg = RandomForestRegressor(criterion='mae', n_estimators=15, max_depth=7, min_samples_split=4, min_samples_leaf=2)
randomTrees3Reg.fit(features_all, labels_all)
pred = randomTrees3Reg.predict(np_test_all)
"""

np_ID = df_test['Id'].values
df_test_answers = pd.DataFrame({'Id': np_ID, 'SalePrice': pred} )
df_test_answers.to_csv("Answers/housing_answers_ExtraTrees_selection.csv", sep=',', index=False)

