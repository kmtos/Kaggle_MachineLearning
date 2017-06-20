import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math

#######################################################
# Given a df, this finds the column and value of
# The best split for the next two leaves based on gini
# tup = (node #, amount of gini decrease, the column name, 
#        the value split at, range between splits)
#######################################################
def FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal):
  columns = [col for col in df if col != className and col != idColumn]
  bestGiniSplit = ( -1, '', -100, -1)
  for col in columns:
    unique = df[col].unique()
    high = unique.max()
    low =  unique.min() 
    splitLength = (high - low) / nGiniSplits
    if len(unique) <= nGiniSplits: #if the number of unique values is less than the desired number of splits, then just make the unique values the splits
      splits = unique
    else: #Find the number of splits for the current column
      splits = []
      for i in range(nGiniSplits):
        splits.append(low + (i * splitLength) )
    bestSplitForCol = CalcBestGiniSplit(df, className, col, splits) #Find the best split for this column
    if bestSplitForCol[0] > bestGiniSplit[0]: #See if this column provides the best Gini decrease so far
      bestGiniSplit = (nodeCount, bestSplitForCol[0], col, bestSplitForCol[1], splitLength )
  if bestGiniSplit[1] < giniEndVal: bestGiniSplit(nodeCount, np.NaN, '', np.NaN, np.NaN)
  return bestGiniSplit
    
#############################################################
# given a column an it's splits, find the best gini decrease
# tup = (amount of gini decrease,the value split at)
#############################################################
def CalcBestGiniSplit(df, className, colName, splits):
  uniqueClasses = df[className].unique()
  bestGiniDecrease = (-100, -1)
  counts = []
  for split in splits:
    counts.append( (len(df[ (df[className] == uniqueClasses[0]) & (df[colName] > split) ] ) , len(df), split) )
    counts.append( (len(df[ (df[className] == uniqueClasses[0]) & (df[colName] < split) ] ) , len(df), split) )
  for tup in counts:
    giniDecrease = tup[0]/tup[1] * (1 - tup[0]/tup[1]) * 2 
    if giniDecrease > bestGiniDecrease[0]:
      bestGiniDecrease = (giniDecrease, tup[2])
  return bestGiniDecrease
  
##################################################################
# Returns a list of tuples describing the test point's membership
# tup = (node number, membership value at node)
##################################################################
def FuzzyMembershipLinear(row, className, giniSplits, alpha):
  membership = []
  count = 0
  for split in giniSplits:
    if row[col] > (split[2] + split[3]*alpha): membership.append( (count, 1) )
    if row[col] < (split[2] - split[3]*alpha): membership.append( (count, 0) )
    else:  membership.append( (count, (row[col] - split[2] - split[3]) / (split[3] * 2) ) )
    count += 1
  return membership

##################################################################
# Makes a Decision Tree that saves output in three forms:
#    1) The tuple found with FindingBestSplit for each node
#    2) The node number and the DF ID's at that leaf
#    3) The Decision that was made at each leaf
##################################################################
def MakeTree(df, className, nGiniSplits, alpha, giniEndVal, maxDepth, idColumn, outputDir):
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodesAndValues = [] #Node number,  amount of gini decrease, the column name, the value split at, range between splits)
  nodeDFIds = [] #The point ID's that are in each leaf
  nodeCount = 0
  try:
    while nodeCount <= maxNodes: #checking that I haven't met the max node numb set by maxDepth
      if nodeCount == 0: #for the trunk or first tree node
        nodesAndValues.append(FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal) ) 
        nodeDFIds.append( (nodeCount, df[idColumn].tolist()) )
      else:
        parentTup = nodesAndValues[(nodeCount-1) // 2] #getting parent nodesAndValues tuple
        parentDFIDs = nodeDFIds[(nodeCount-1) // 2] #getting parent dataframe row ID's
        if pd.isnull(parentTup[1]) and  parentTup[2] == '' and pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): 
          nodesAndValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
          nodeDFIds.append( (nodeCount, pd.DataFrame()) )
        else:
          if nodeCount % 2  == 1: dfCurr = df.loc[(df[className].isin(parentDFIDs[1])) & (df[parentTup[2]] < parentTup[3]) ] #Getting dataframe elements that are lower than the split
          else: dfCurr = df.loc[(df[className].isin(parentDFIDs[1])) & (df[parentTup[2]] >= parentTup[3]) ] #getting dataframe elements that are greater than or equal to the split
          nodesAndValues.append(FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal) )
          nodeDFIds.append( (nodeCount, dfCurr[idColumn].tolist()) )
      nodeCount += 1

    #Writing out the tuples for the nodes and cuts on which columns and dataframe ID's in each of the leaves
    nodeAndValueFileName = outputDir + "DoneTree_maxDepth" + str(maxDepth) + "_nSplits" + str(nGiniSplits) + "_alpha" + str(alpha) + "giniEndVal" + str(giniEndVal) + "_nodesAndValues.csv"
    nodeAndValueFile = open(nodeAndValueFileName, 'w')
    nodeAndValueFileCSV=csv.writer(nodeAndValueFile)
    for tup in nodesAndValues:
      nodeAndValueFileCSV.writerow(tup)
    nodeDFIDsFileName = outputDir + "DoneTree_maxDepth" + str(maxDepth) + "_nSplits" + str(nGiniSplits) + "_alpha" + str(alpha) + "giniEndVal" + str(giniEndVal) + "_nodeDFIds.csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)
    
    #Finding and writing the decisions of the leaves
    nodeDecisions = []
    minNodeNotLeaf = maxNodes
    for ite in range(maxNodes, maxNodes - 2**maxDepth, -1):
      index = ite
      currentLeaf = nodesAndValues[index]
      currentDF = df.loc[df[className].isin(nodeDFIds[index][1])]
      while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]):
        index = (ite-1) // 2
        currentLeaf = nodesAndValues[index]
        currentDF = df.loc[df[className].isin(nodeDFIds[index][1])]
      maxCount = -100000000000000000000000000000000
      maxClassVal = -100000000000000000000000000000000
      print ("ite=", ite, "\tcurrentLeaf=", currentLeaf, currentDF[className].value_counts().to_dict())
      for classVal, row in  currentDF[className].value_counts().to_dict().items():
        print ("classVal=", classVal, "\trow=", int(row), "\ttype=", type(int(row) ) )
        if int(row) > maxCount:
          print ("\tmaxCount=", maxCount, "\tmaxClassVal=", maxClassVal)
          maxCount = int(row)
          maxClassVal = classVal
          print ("\tmaxCount=", maxCount, "\tmaxClassVal=", maxClassVal)
      listTEST = [ ('a',1), ('b',2), ('c',3) ]
      try: 
        next(tup for tup in nodeDecisions if tup[0] == index)
        print ("Node already added from brother leaf being null from hitting 'giniEndVal', i.e. leaf can't be improved anymore")
      except StopIteration: 
        nodeDecisions.append( (index, maxClassVal) )
         
    nodeDecisionsFileName = outputDir + "DoneTree_maxDepth" + str(maxDepth) + "_nSplits" + str(nGiniSplits) + "_alpha" + str(alpha) + "giniEndVal" + str(giniEndVal) + "_nodeDecisions.csv"
    nodeDecisionsFile = open(nodeDecisionsFileName, 'w')
    nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
    for tup in nodeDecisions:
      nodeDecisionsFileCSV.writerow(tup)

             
  except KeyboardInterrupt:
    nodeAndValueFileName = outputDir + "LastCompleteNode" + str(nodeCount-1) + "_nodesAndValues.csv"
    nodeAndValueFile = open(nodeAndValueFileName, 'w')
    nodeAndValueFileCSV=csv.writer(nodeAndValueFile)
    for tup in nodesAndValues:
      nodeAndValueFileCSV.writerow(tup)
    nodeDFIDsFileName = outputDir + "LastCompleteNode" + str(nodeCount-1) + "_nodeDFIds.csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)

    
 
