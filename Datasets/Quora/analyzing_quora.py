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
#df_train['diffWords'] = df_train['diffWords'].abs()
#df_train['diffChars'] = df_train['diffChars'].abs()
#df_train.to_csv("training_data/train_match_final.csv", index=False)

################################################
# Getting numeric Data and getting numpy splits
################################################
print (df_train.dtypes )
df_train_noStrings = df_train._get_numeric_data()
print (df_train_noStrings.dtypes )

df_train_noStrings = df_train_noStrings.drop( ['id', 'qid1', 'qid2'], axis=1 )
print (df_train_noStrings.dtypes )

from sklearn.model_selection import train_test_split
np_train_features_noStrings = df_train_noStrings.values
labels_train, features_train = targetFeatureSplit(np_train_features_noStrings )
features_train_1, features_test_1, labels_train_1, labels_test_1 = train_test_split(features_train, labels_train, test_size=0.25, random_state=42)


df_train_noStrings_noStopOnly = df_train_noStrings.drop( ['match_stem', 'match_stem_double', 'match_stem_triple', 'nChars_match_stem', 'nChars_match_stem_double', 'nChars_match_stem_triple', 
                                                          'sequential_match_stem', 'sequential_match_stem_double', 'sequential_match_stem_triple'], axis=1)
np_train_features_noStrings_noStopOnly = df_train_noStrings_noStopOnly.values
labels_train_noStopOnly, features_train_noStopOnly = targetFeatureSplit(np_train_features_noStrings_noStopOnly )
features_train_noStopOnly_1, features_test_noStopOnly_1, labels_train_noStopOnly_1, labels_test_noStopOnly_1 = train_test_split(features_train_noStopOnly, labels_train_noStopOnly, 
                                                                                                                                test_size=0.25, random_state=42)
"""
####################
# Data Searching
####################
#    Visualization
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from collections import defaultdict

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 300) # viridis
cax = ax1.imshow(df_train_noStrings.corr(), interpolation="nearest", cmap=cmap)
ax1.grid(True, color='grey')
plt.title('Quora Correlation of Features', y=1.1, size=15)
labels = [column for column in df_train_noStrings]
ax1.set_xticks(np.arange(len(labels))-.5)
ax1.set_xticklabels(labels,fontsize=6, rotation=45, ha='right')
ax1.set_yticks(np.arange(len(labels))-.5)
ax1.set_yticklabels(labels,fontsize=6, rotation=45, va='top')
ax1.set_xlim(16.5, -.5)
ax1.set_ylim(16.5, -.5)
# Add colorbar, make sure to specify tick locations to match desired ticklabels
fig.colorbar(cax, ticks=[1, .75, .5, .25, 0, -.25, -.5, -.75, -1])
ite = 0
for i in df_train_noStrings.corr():
  jte = 0
  for j in df_train_noStrings.corr():
    ax1.annotate(round(df_train_noStrings.corr()[i][j], 2), (ite+.35,jte) )
    jte += 1
  ite += 1
plt.show()

############################################################################
# Showing the feature importances before and after making choices on values
############################################################################

visualizeDecisionTree(df_train_noStrings, 'Quora FeatCorr')
maxLabel = findMaxCorr(df_train_noStrings, 'is_duplicate')
newDfs = {}
newDfs['0'] = df_train_noStrings[df_train_noStrings[maxLabel] == 0]
newDfs['>0'] = df_train_noStrings[df_train_noStrings[maxLabel] > 0]

print ("########################\n## Feature importances before and after choices\n######################")
for k,dfIte in newDfs.items():
  print ("\n\nk=", k, "\n", dfIte.corr()) 
  clfIte = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=10, min_samples_leaf=2, class_weight=None, n_estimators=10 )
  dfIte = dfIte.drop([maxLabel], axis=1)
  np_train_features_noStrings_ite = dfIte.values
  labels_train_ite, features_train_ite = targetFeatureSplit(np_train_features_noStrings_ite )
  clfIte.fit(features_train_ite, labels_train_ite)
  visualizeDecisionTree(dfIte, 'Quora FeatCorr: no ' + maxLabel + k)

  maxLabel_1 = findMaxCorr(dfIte, 'is_duplicate')
  newDfs_1 = {}
  newDfs_1['0'] = dfIte[dfIte[maxLabel_1] == 0]
  newDfs_1['>0'] = dfIte[dfIte[maxLabel_1] > 0]
  for k_1,dfIte_1 in newDfs_1.items():
    print ("\n\n  k_1=", k_1, "\n", dfIte_1.corr()) 
    clfIte = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=10, min_samples_leaf=2, class_weight=None, n_estimators=10 )
    dfIte_1 = dfIte_1.drop([maxLabel_1], axis=1)
    np_train_features_noStrings_ite_1 = dfIte_1.values
    labels_train_ite_1, features_train_ite_1 = targetFeatureSplit(np_train_features_noStrings_ite_1 )
    clfIte.fit(features_train_ite_1, labels_train_ite_1)
    visualizeDecisionTree(dfIte_1, 'Quora FeatCorr: no ' + maxLabel + k + ", " + maxLabel_1 + k_1)

    maxLabel_2 = findMaxCorr(dfIte_1, 'is_duplicate')
    newDfs_2 = {}
    newDfs_2['0'] = dfIte_1[dfIte_1[maxLabel_2] == 0]
    newDfs_2['>0'] = dfIte_1[dfIte_1[maxLabel_2] > 0]
    for k_2,dfIte_2 in newDfs_2.items():
      print ("    k_2=", k_2)
      clfIte = RandomForestClassifier(criterion='entropy', max_depth=5, min_samples_split=10, min_samples_leaf=2, class_weight=None, n_estimators=10 )
      dfIte_2 = dfIte_2.drop([maxLabel_2], axis=1)
      np_train_features_noStrings_ite_2 = dfIte_2.values
      labels_train_ite_2, features_train_ite_2 = targetFeatureSplit(np_train_features_noStrings_ite_2 )
      clfIte.fit(features_train_ite_2, labels_train_ite_2)
      visualizeDecisionTree(dfIte_2, 'Quora FeatCorr: no ' + maxLabel + k + ", " + maxLabel_1 + k_1 + ", " + maxLabel_2 + k_2)
      maxLabel_3 = findMaxCorr(dfIte_2, 'is_duplicate')

# Export our trained model as a .dot file
from sklearn import tree
from inspect import getmembers
labels = []
for col in df_train_noStrings:
  labels.append(col)

iTree = 0
clf = RandomForestClassifier(criterion='entropy', max_depth=3, min_samples_leaf=1, min_samples_split=5, class_weight=None, n_estimators=10, oob_score=True, bootstrap=True, random_state=1 )
for tree_in_forest in clf.estimators_:
    with open('TreeOutput/tree_' + str(iTree) + '_max_depth5.dot', 'w') as my_file:
        my_file = tree.export_graphviz(tree_in_forest, out_file=my_file, max_depth=5, impurity=True, feature_names=list(df_train_noStrings), 
                                       class_names=['Different', 'Duplicate'], rounded=True, filled=True)
    print ("Tree #", iTree)
    for i in range(len(labels) ):
      print ("\t", labels[i], tree_in_forest.feature_importances_[i] )
    iTree = iTree + 1
    #Convert .dot to .png to allow display in web notebook
for ite in range(iTree):
    check_call(["dot","-Tpng", 'TreeOutput/tree_' + str(ite) + '_max_depth5.dot', "-o",'TreeOutput/tree_' + str(ite) + '_max_depth5.png'])

# Plots of the features

diffWords                          
diffChars                          
match_stem                         
match_stem_noStop                  
match_stem_double                  
match_stem_double_noStop           
match_stem_triple                  
match_stem_triple_noStop           
nChars_match_stem                  
nChars_match_stem_noStop           
nChars_match_stem_double           
nChars_match_stem_double_noStop    
nChars_match_stem_triple           
nChars_match_stem_triple_noStop    
sequential_match_stem              
sequential_match_stem_noStop       
sequential_match_stem_double       
sequential_match_stem_double_noStop
sequential_match_stem_triple       
sequential_match_stem_triple_noStop

pairs = []
df_train_dup = df_train_noStrings[df_train_noStrings['is_duplicate'] == 1 ]
df_train_nodup = df_train_noStrings[df_train_noStrings['is_duplicate'] == 0 ]
pairs.append(['nChars_match_stem', 'diffWords'])
pairs.append(['nChars_match_stem_noStop', 'diffWords'])
pairs.append(['nChars_match_stem_double', 'diffWords'])
pairs.append(['nChars_match_stem_double_noStop', 'diffWords'])
pairs.append(['nChars_match_stem_triple', 'diffWords'])
pairs.append(['nChars_match_stem_triple_noStop', 'diffWords'])
pairs.append(['sequential_match_stem_triple', 'diffWords'])
pairs.append(['sequential_match_stem_triple_noStop', 'diffWords'])
pairs.append(['sequential_match_stem_double_noStop', 'diffWords'])
plotGridOf2DsWithColor(df_train_dup, pairs, ["blue"], [ "o"], "matches_and_sequentialMatch_vs_diffwords.png", "png" , 0.01, 55, 1) #55 was 110
plotGridOf2DsWithColor(df_train_nodup, pairs, ["red"], [ "o"], "matches_and_sequentialMatch_vs_diffwords.png", "png" , 0.01, 55, 1) #55 was 110
#plotGridOf2DsWithColor(df_train_nodup, pairs, ["blue", "red"], [ "o", "s"], "matches_and_sequentialMatch_vs_diffwords.png", "png" , 0.01, 55, 1) #55 was 110

labels = ["match_stem","match_stem_double", "match_stem_triple", "nChars_match_stem", "nChars_match_stem_double", "nChars_match_stem_triple", "sequential_match_stem", "sequential_match_stem_double", "sequential_match_stem_triple"]

labels_noStop = ["match_stem_noStop","match_stem_double_noStop", "match_stem_triple_noStop", "nChars_match_stem_noStop", "nChars_match_stem_double_noStop", "nChars_match_stem_triple_noStop", "sequential_match_stem_noStop", "sequential_match_stem_double_noStop", "sequential_match_stem_triple_noStop"]

df_train_dup = df_train_noStrings[df_train_noStrings['is_duplicate'] == 1 ]
df_train_nodup = df_train_noStrings[df_train_noStrings['is_duplicate'] == 0 ]
fig = plt.figure()
for i in range(len(labels)):
  ax = fig.add_subplot(3, 3, i+1)
  print ("i=", labels[i])
  if "triple" in labels[i]:
    print ("Triple yLimSet")
    ax.set_ylim(0,7500)
  if "double" in labels[i]:
    print ("DOUBLE yLim Set")
    ax.set_ylim(0,8000)
  df_train_dup[labels[i] ].hist(bins=100, ax=ax, alpha=0.5)
  df_train_nodup[labels[i] ].hist(bins=100, ax=ax, alpha=0.5)
  ax.set_title(labels[i] )
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
fig.tight_layout()
plt.show()
"""

###################
# Using algorithms
###################
############# Trying kNearest-neighbors ############
clf1 = KNeighborsClassifier(n_neighbors=10, weights=distanceWeight, metric='euclidean', p=5)
clf1.fit(features_train_1, labels_train_1)
print ("clf1=", clf1.score(features_test_1, labels_test_1) )

clf2 = KNeighborsClassifier(n_neighbors=50, weights=distanceWeight, metric='euclidean', p=5)
clf2.fit(features_train_1, labels_train_1)
print ("clf2=", clf2.score(features_test_1, labels_test_1) )

clf3 = KNeighborsClassifier(n_neighbors=100, weights=distanceWeight, metric='euclidean', p=5)
clf3.fit(features_train_1, labels_train_1)
print ("clf3=", clf3.score(features_test_1, labels_test_1) )

clf4 = KNeighborsClassifier(n_neighbors=150, weights=distanceWeight, metric='euclidean', p=5)
clf4.fit(features_train_1, labels_train_1)
print ("clf4=", clf4.score(features_test_1, labels_test_1) )

clf5 = KNeighborsClassifier(n_neighbors=200, weights=distanceWeight, metric='euclidean', p=5)
clf5.fit(features_train_1, labels_train_1)
print ("clf5=", clf5.score(features_test_1, labels_test_1) )

clf6 = KNeighborsClassifier(n_neighbors=250, weights=distanceWeight, metric='euclidean', p=5)
clf6.fit(features_train_1, labels_train_1)
print ("clf6=", clf6.score(features_test_1, labels_test_1) )

clf7 = KNeighborsClassifier(n_neighbors=300, weights=distanceWeight, metric='euclidean', p=5)
clf7.fit(features_train_1, labels_train_1)
print ("clf7=", clf7.score(features_test_1, labels_test_1) )

clf8 = KNeighborsClassifier(n_neighbors=350, weights=distanceWeight, metric='euclidean', p=5)
clf8.fit(features_train_1, labels_train_1)
print ("clf8=", clf8.score(features_test_1, labels_test_1) )

clf9 = KNeighborsClassifier(n_neighbors=400, weights=distanceWeight, metric='euclidean', p=5)
clf9.fit(features_train_1, labels_train_1)
print ("clf9=", clf9.score(features_test_1, labels_test_1) )

clf10 = KNeighborsClassifier(n_neighbors=450, weights=distanceWeight, metric='euclidean', p=5)
clf10.fit(features_train_1, labels_train_1)
print ("clf10=", clf10.score(features_test_1, labels_test_1) )

clf11 = KNeighborsClassifier(n_neighbors=500, weights=distanceWeight, metric='euclidean', p=5)
clf11.fit(features_train_1, labels_train_1)
print ("clf11=", clf11.score(features_test_1, labels_test_1) )

clf12 = KNeighborsClassifier(n_neighbors=550, weights=distanceWeight, metric='euclidean', p=5)
clf12.fit(features_train_1, labels_train_1)
print ("clf12=", clf12.score(features_test_1, labels_test_1) )

clf13 = KNeighborsClassifier(n_neighbors=600, weights=distanceWeight, metric='euclidean', p=5)
clf13.fit(features_train_1, labels_train_1)
print ("clf13=", clf13.score(features_test_1, labels_test_1) )

clf14 = KNeighborsClassifier(n_neighbors=650, weights=distanceWeight, metric='euclidean', p=5)
clf14.fit(features_train_1, labels_train_1)
print ("clf14=", clf14.score(features_test_1, labels_test_1) )

clf15 = KNeighborsClassifier(n_neighbors=700, weights=distanceWeight, metric='euclidean', p=5)
clf15.fit(features_train_1, labels_train_1)
print ("clf15=", clf15.score(features_test_1, labels_test_1) )

"""
paramDictKNN = {'n_neighbors': [10,25,50,75,100], 'weights':['distance', distanceWeight], 'metric':['euclidean', 'minkowski'], 'p':[2,3,4, 5]}
knn = KNeighborsClassifier()
print ('\n##########################################\n GridSearchCV with k-Nearest Neighbors:')
kNNBestParams = gridSearch(knn, features=features_train, labels=labels_train, parameters=paramDictKNN, cross_validation=10)

#############  Training Random Forest ##############
clf1 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf1.fit(features_train_1, labels_train_1)
print ("clf1=", clf1.score(features_test_1, labels_test_1), "\toob_score=", clf1.oob_score_ )

clf2 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf2.fit(features_train_1, labels_train_1)
print ("clf2=",  clf2.score(features_test_1, labels_test_1), "\toob_score=", clf2.oob_score_ )

clf3 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf3.fit(features_train_1, labels_train_1)
print ("clf3=", clf3.score(features_test_1, labels_test_1), "\toob_score=", clf3.oob_score_ )

clf4 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf4.fit(features_train_1, labels_train_1)
print ("clf4=", clf4.score(features_test_1, labels_test_1), "\toob_score=", clf4.oob_score_ )

clf5 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf5.fit(features_train_1, labels_train_1)
print("clf5=", clf5.score(features_test_1, labels_test_1), "\toob_score=", clf5.oob_score_ )

clf6 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf6.fit(features_train_1, labels_train_1)
print ("clf6=", clf6.score(features_test_1, labels_test_1) , "\toob_score=", clf6.oob_score_)

clf7 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf7.fit(features_train_1, labels_train_1)
print("clf7=", clf7.score(features_test_1, labels_test_1) , "\toob_score=", clf7.oob_score_)

clf8 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf8.fit(features_train_1, labels_train_1)
print ("clf8=", clf8.score(features_test_1, labels_test_1) , "\toob_score=", clf8.oob_score_)

clf9 = RandomForestClassifier(criterion='entropy', max_depth=41, min_samples_leaf=1, min_samples_split=10, class_weight=None, n_estimators=1000, oob_score=True, bootstrap=True, random_state=1 )
clf9.fit(features_train_1, labels_train_1)
print ("clf9=", clf9.score(features_test_1, labels_test_1) , "\toob_score=", clf9.oob_score_)
"""
