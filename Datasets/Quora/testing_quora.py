import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import csv
import sys
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *
from string_data_manipulation import *
import math
pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/')

from sklearn.ensemble import RandomForestClassifier
from KNearestNeighbors import *
from GridSearch import *

df_train =  pd.read_csv('training_data/train_match_final.csv', header=0)
df_test = pd.read_csv('testing_data/test_match_final.csv', header=0)
################################################
# Getting numeric Data and getting numpy splits
################################################
df_train_noStrings = df_train._get_numeric_data()
print (df_test.dtypes )

df_train_noStrings = df_train_noStrings.drop( ['id', 'qid1', 'qid2'], axis=1 )

from sklearn.model_selection import train_test_split
np_train_features_noStrings = df_train_noStrings.values
labels_train, features_train = targetFeatureSplit(np_train_features_noStrings )


df_train_noStrings_noStopOnly = df_train_noStrings.drop( ['match_stem', 'match_stem_double', 'match_stem_triple', 'nChars_match_stem', 'nChars_match_stem_double', 'nChars_match_stem_triple', 'sequential_match_stem', 'sequential_match_stem_double', 'sequential_match_stem_triple'], axis=1)
np_train_features_noStrings_noStopOnly = df_train_noStrings_noStopOnly.values
labels_train_noStopOnly, features_train_noStopOnly = targetFeatureSplit(np_train_features_noStrings_noStopOnly )

np_testID = df_test['test_id'].values
df_test_noStrings = df_test._get_numeric_data()
np_test_features_noStrings = df_test_noStrings.values
labels_test, features_test = targetFeatureSplit(np_test_features_noStrings )

print ("TRAIN\n", df_train_noStrings.dtypes )
print ("\n\nTEST", df_test_noStrings.dtypes )
###################
# Using algorithms
###################

"""
#############  Training Random Forest ##############
clf1 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=10, oob_score=True, bootstrap=True, random_state=1 )
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
pred1 = pred1.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred1} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst10", sep=',', index=False)
print ("DONE 10")

clf2 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=50, oob_score=True, bootstrap=True, random_state=1 )
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
pred2 = pred2.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred2} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst50", sep=',', index=False)
print ("DONE 50")

clf3 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=100, oob_score=True, bootstrap=True, random_state=1 )
clf3.fit(features_train, labels_train)
pred3 = clf3.predict(features_test)
pred3 = pred3.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred3} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst100", sep=',', index=False)
print ("DONE 100")
"""
clf4 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=200, oob_score=True, bootstrap=True, random_state=1 )
clf4.fit(features_train, labels_train)
pred4 = clf4.predict_proba(features_test)
print ("############\nPREDICTIONS\n################", pred4)
print ("############\nPREDICTIONS[1]\n################", pred4[:,1])
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred4[:,1].tolist()} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst200_PredProb.csv", sep=',', index=False)
print ("DONE 250")
"""
clf5 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=500, oob_score=True, bootstrap=True, random_state=1 )
clf5.fit(features_train, labels_train)
pred5 = clf5.predict(features_test)
pred5 = pred5.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred5} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst500", sep=',', index=False)
print ("DONE 500")

clf6 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=750, oob_score=True, bootstrap=True, random_state=1 )
clf6.fit(features_train, labels_train)
pred6 = clf6.predict(features_test)
pred6 = pred6.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred6} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst750", sep=',', index=False)
print ("DONE 750")

clf7 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf7.fit(features_train, labels_train)
pred7 = clf7.predict(features_test)
pred7 = pred7.astype(int)
df_answers = pd.DataFrame({'id': np_testID, 'is_duplicate': pred7} )
df_answers.to_csv("Answers/Quora_RF_dep41_minSamp10_nEst1000", sep=',', index=False)
print ("DONE 1000")
"""
