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
def plotGridOf2DsWithColor(features, pairs, colors, markerShapes, plotName, fileTypeSuffix, alphaValue, size):
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
      plt.scatter(dfList[ite].ix[:,0] + xOffSet, dfList[ite].ix[:,1] + yOffSet, color = colors[ite], marker=markerShapes[ite], alpha=alphaValue, s=110, label=legendLabel)
      # The rest of the for loop is for mamking the legend. It's complicated because it's difficult to get the alpha of a scatterplot different on the legend. Had to make lines to do it.
      globals()['line%s' % ite] = pylab.Line2D( range(1), range(1), color='white', marker=markerShapes[ite], markersize=10, markerfacecolor=colors[ite], alpha=1) 
      lineTuple.append(globals()['line%s' % ite] )
      lineNameTuple.append(legendLabel)
    plt.grid(True)
  leg = plt.legend(tuple(lineTuple), tuple(lineNameTuple), bbox_to_anchor=(1.0, -1), loc=4, fontsize=8)
  plt.savefig(plotName, format=fileTypeSuffix)
  plt.show()
