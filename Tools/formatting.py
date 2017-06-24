import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math
"""
given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list (this should be the  quantity you want to predict)
return targets and features as separate lists. (sklearn can generally handle both lists and numpy arrays as input formats when training/predicting)
"""
def targetFeatureSplit( data ):
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )
    return target, features

"""
Takes features and pairs and plots 2D plots with colors of the pairs. 
features should be a dataframe, with the first column being the predicted label (Survived for the Titanic Dataset).
pairs should be a list of arrays with the two values to be plotted together.
colors should have as many values as unique label values.
"""
def plotGridOf2DsWithColor(features, pairs, colors, markerShapes, plotName, fileTypeSuffix, alphaValue, size, offSetMulti):
  if len(features.ix[:,0].unique() ) != len(colors):
    print ("Numer of colors=", len(colors), "\tNumber of unique features=", len(features.ix[:,0].unique() ) )
    print ("\nNumber of colors does not correspond to number of unique label values. Please fix!")
    return
  np_pairs = np.array(pairs)
  for i in range(np_pairs.shape[0]):
    print("Plot", i , pairs[i])
    xUnique = len(features[pairs[i][0]].unique() )*1.0
    yUnique = len(features[pairs[i][1]].unique() )*1.0
    labelsUnique = features.ix[:,0].unique().tolist()
    plt.subplot(math.ceil(math.sqrt(np_pairs.shape[0]) ), math.ceil(math.sqrt(np_pairs.shape[0]) ), i+1) # This is setting the subplot grid
    plt.xlabel(np_pairs[i][0])
    plt.ylabel(np_pairs[i][1])
    # dfList is a list of dataframes of the columns in the "pairs" to be plotted, where each dataframe has a unique features in the predictive feature (Survived for Titanic dataset).
    dfList = []
    for label in features.ix[:,0].unique():
      dfList.append(features.loc[ features.ix[:,0] == label, [ pairs[i][0], pairs[i][1] ] ] )
    emptyTuple = ()
    lineTuple = list(emptyTuple)
    lineNameTuple = list(emptyTuple)
    for ite in range(len(dfList) ):
      # Offsets are for having part of the point being able to be seen around the point itself. By dividing by unique values, it makes this do nothing for continuous variables
      xOffSet = .05 * math.cos(2*math.pi/ (1.0 * (1+ite) * len(labelsUnique) ) ) / xUnique
      yOffSet = .05 * math.sin(2*math.pi/ (1.0 * (1+ite) * len(labelsUnique) ) ) / yUnique
      legendLabel = features.columns[0] + '=' + str(labelsUnique[ite] )
      plt.scatter(dfList[ite].ix[:,0] + xOffSet*offSetMulti, dfList[ite].ix[:,1] + yOffSet*offSetMulti, color = colors[ite], marker=markerShapes[ite], alpha=alphaValue, s=size, label=legendLabel)#s=110
      # The rest of the for loop is for mamking the legend. It's complicated because it's difficult to get the alpha of a scatterplot different on the legend. Had to make lines to do it.
      globals()['line%s' % ite] = pylab.Line2D( range(1), range(1), color='white', marker=markerShapes[ite], markersize=4, markerfacecolor=colors[ite], alpha=1) 
      lineTuple.append(globals()['line%s' % ite] )
      lineNameTuple.append(legendLabel)
    plt.grid(True)
  leg = plt.legend(tuple(lineTuple), tuple(lineNameTuple), bbox_to_anchor=(0, 0), loc=0, fontsize=8)
  plt.savefig(plotName, format=fileTypeSuffix)
  plt.show()

#####################################################################
# For Writing out true and false positives and negatives or a sample
#####################################################################
def printTrueFalsePosNeg(labels_test=None, pred=None):
  if labels_test is None or pred is None:
    print ("You gave me nothing to compare and print, numbnut.")
    return
  if len(labels_test) != len(pred):
    print ("The lengths of the test and predicted aren't the same. Soemthing is wrong.")
  sumTruePositives = 0
  sumTrueNegatives = 0
  sumFalseNegative = 0
  sumFalsePositve = 0
  for i in range(len(pred)):
    iTrue = labels_test[i]
    iPred = pred[i]
    if iPred == iTrue and iPred == 1:
      sumTruePositives += 1
    if iPred == iTrue and iPred == 0:
      sumTrueNegatives += 1
    if iPred != iTrue and iTrue == 1:
      sumFalseNegative += 1
    if iPred != iTrue and iPred == 1:
      sumFalsePositve += 1
  print ("sumTruePositives= " , sumTruePositives , "\tsumTrueNegatives= " , sumTrueNegatives, "\tsumFalseNegative= " , sumFalseNegative , "\tsumFalsePositve= " , sumFalsePositve)


#########################################################################################
# Calculating Gini Imipurity decrease for 2 classes that are 0 and 1
# Returns teh giniImpDecreaseas well as a List of the Imp for values of otherColumnName
#########################################################################################
def calcGiniImp(df, labelColumnName, otherColumnName):
  giniImpList = []
  giniImpListTup = []
  sumList = []
  countList = []
  giiniImp = 1 
  totalCount = 0
  totalSum = 0
  weightList = []
  
  labels = df[labelColumnName].unique()
  if len(labels) != 2:
    print ("This only works for two class labels!!!")
    return -1
  df_agg = df[[otherColumnName, labelColumnName]].groupby([otherColumnName], as_index=False).aggregate(['sum', 'mean', 'count']) 
  countLabel = df_agg[labelColumnName, 'count']
  sumLabel = df_agg[labelColumnName, 'sum']
  for i in df[otherColumnName].unique():
    countList.append((i, countLabel[i]) )
    sumList.append(sumLabel[i] )
  for i in range(len(countList) ):
    totalCount += countList[i][1]
    totalSum += sumList[i]
  giniImpDecrease = (totalSum/totalCount * (1 - totalSum/totalCount) * 2)

  for i in range(len(countList) ):
    print ("countList=", countList[i])
    print ("sumList=", sumList[i] )
    if countList[i][1] == 0:
      giniImpList.append(0)
      weightList.append(0)
    else:
      giniImpList.append( sumList[i]/countList[i][1] * (1 - sumList[i]/countList[i][1]) *2 )
      giniImpListTup.append( (countList[i][0], sumList[i]/countList[i][1] * (1 - sumList[i]/countList[i][1]) *2) )
      weightList.append(countList[i][1] / totalCount )
  for i in  range(len(giniImpList) ):
    giniImpDecrease -= giniImpList[i] * weightList[i]
  print ("For", otherColumnName, " the Decrease is", giniImpDecrease, "and the full output is", giniImpListTup)
  return giniImpDecrease, giniImpListTup


###################################
#    Visualization of Decison Tree
###################################
def visualizeDecisionTree(df, titleName):
  from matplotlib import cm as cm
  import matplotlib.pyplot as plt
  from collections import defaultdict
  
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  cmap = cm.get_cmap('jet', 300) # viridis
  cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
  ax1.grid(True, color='grey')
  plt.title(titleName, y=1.1, size=15)
  labels = [column for column in df]
  ax1.set_xticks(np.arange(len(labels))-.5)
  ax1.set_xticklabels(labels,fontsize=6, rotation=45, ha='right')
  ax1.set_yticks(np.arange(len(labels))-.5)
  ax1.set_yticklabels(labels,fontsize=6, rotation=45, va='top')
  ax1.set_xlim(len(df.columns)-.5, -.5)
  ax1.set_ylim(len(df.columns)-.5, -.5)
  # Add colorbar, make sure to specify tick locations to match desired ticklabels
  fig.colorbar(cax, ticks=[1, .75, .5, .25, 0, -.25, -.5, -.75, -1])
  ite = 0
  for i in df.corr():
    jte = 0
    for j in df.corr():
      ax1.annotate(round(df.corr()[i][j], 2), (ite+.35,jte) )
      jte += 1
    ite += 1
  plt.show()

def findMaxCorr(df, predictedValue):
  coef = df.corr()
  maxRel = -1
  maxLabel = ""
  labels = []
  for col in coef:
    labels.append(col)
  ite = 0
  for row in coef[predictedValue]:
    if abs(row) > maxRel and abs(row) != 1.0:
      maxRel = abs(row)
      maxLabel = labels[ite]
    ite += 1
  print ("maxRel=", maxRel, "\tmaxLabel=", maxLabel)
  return maxLabel

