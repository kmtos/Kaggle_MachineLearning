import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math
from operator import itemgetter


#######################################################
# Given a df, this finds the column and value of
# The best split for the next two leaves based on gini
# tup = (node #, amount of gini decrease, the column name, 
#        the value split at, range between splits)
#######################################################
def FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal):
  columns = [col for col in df if col != className and col != idColumn]
  bestGiniSplit = (-1, -1, '', -100, -1)
  for classVal, rows in df[className].value_counts().to_dict().items():
    if len(df) == rows: 
      print ("*****************END NODE")
      return ( nodeCount, 1.0, 'ThisIsAnEndNode', np.NaN, np.NaN)
  for col in columns:
    useUnique = False
    unique = sorted(df[col].unique() )
    high = unique[len(unique)-1]
    low =  unique[0]
    splitLength = (high - low) / (nGiniSplits+1)
    splits = []
    if len(unique) <= nGiniSplits: #if the number of unique values is less than the desired number of splits, then just make the unique values the splits
      for i in range(len(unique)-1):
        splits.append( (unique[i] + unique[i+1])/2.0)
    else: #Find the number of splits for the current column
      for i in range(1,nGiniSplits+1):
        splits.append(low + (i * splitLength) )
    bestSplitForCol = CalcBestGiniSplit(df, className, col, splits) #Find the best split for this column
    if bestSplitForCol[0] > bestGiniSplit[1]: #See if this column provides the best Gini decrease so far
      bestGiniSplit = (nodeCount, bestSplitForCol[0], col, bestSplitForCol[1], splitLength )
  if bestGiniSplit[1] < giniEndVal: bestGiniSplit = (nodeCount, np.NaN, '', np.NaN, np.NaN)
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
    counts.append( (len(df[ (df[className]==uniqueClasses[0]) & (df[colName]>split) ]), len(df[df[colName]>split]), split,
                    len(df[ (df[className]==uniqueClasses[0]) & (df[colName]<=split) ]), len(df[df[colName]<=split])) )
  for tup in counts:
    giniDecreaseGT = tup[0]/tup[1] * tup[0]/tup[1] +  (1 - tup[0]/tup[1]) * (1 - tup[0]/tup[1])
    giniDecreaseLT = tup[3]/tup[4] * tup[3]/tup[4] +  (1 - tup[3]/tup[4]) * (1 - tup[3]/tup[4])
    giniDecrease = giniDecreaseGT*tup[1]/(tup[1]+tup[4]) + giniDecreaseLT*tup[4]/(tup[1]+tup[4])
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
def MakeTree(df, className, nGiniSplits, alpha, giniEndVal, maxDepth, idColumn, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName):
  print ("###################################\n Making a Decision Tree\n###################################")
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodeValues = [] #Node number,  amount of gini decrease, the column name, the value split at, range between splits)
  nodeDFIds = [] #The point ID's that are in each leaf
  nodeCount = 0
  try:
    while nodeCount <= maxNodes: #checking that I haven't met the max node numb set by maxDepth
      if nodeCount == 0: #for the trunk or first tree node
        nodeValues.append(FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal) ) 
        nodeDFIds.append( (nodeCount, df[idColumn].tolist()) )
      else:
        parentTup = nodeValues[(nodeCount-1) // 2] #getting parent nodeValues tuple
        parentDFIDs = nodeDFIds[(nodeCount-1) // 2][1] #getting parent dataframe row ID's
        print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup, "\tlen(parentDFIDs)=", len(parentDFIDs))
        if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): 
          nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
          nodeDFIds.append( (nodeCount, pd.DataFrame()) )
        else:
          if nodeCount % 2  == 1: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] <= parentTup[3]) ] #Getting dataframe elements that are lower than the split
          else: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] > parentTup[3]) ] #getting dataframe elements that are greater than or equal to the split
          print ("len(dfCurr)=", len(dfCurr), "\tparentTup[3]=", parentTup[3] )
          nodeValues.append(FindingBestSplit(dfCurr, className, idColumn, nGiniSplits, nodeCount, giniEndVal) )
          nodeDFIds.append( (nodeCount, dfCurr[idColumn].tolist()) )
          print ("######## NEW ########:", "nodeValues[nodeCount]=", nodeValues[nodeCount], "\tlen(nodeDFIds[1])=", len(nodeDFIds[1]))
          if not pd.isnull(nodeValues[nodeCount][3]): print ("len(lessThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] <= nodeValues[nodeCount][3]]), "\tlen(greaterThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] > nodeValues[nodeCount][3]]) ) 
      nodeCount += 1

    #Writing out the tuples for the nodes and cuts on which columns and dataframe ID's in each of the leaves
    print ("###########################\nWriting Out the Nodes, Values, and df ID's of the nodes\n###########################")
    nodeValuesFileName = nodeValuesFileName + ".csv"
    nodeValuesFile = open(nodeValuesFileName, 'w')
    nodeValuesFileCSV=csv.writer(nodeValuesFile)
    for tup in nodeValues:
      nodeValuesFileCSV.writerow(tup)
    nodeDFIDsFileName = nodeDFIDsFileName + ".csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)
    
    #Getting the first non-dead leaf, i.e. leaf who's parent has a gini increase greater than the minimum for a leaf to end
    nodeDecisions = []
    minNodeNotLeaf = maxNodes
    for ite in range(maxNodes, maxNodes - 2**maxDepth, -1):
      index = ite 
      currentLeaf = nodeValues[index]
      currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
      while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]):
        index = (ite-1) // 2
        currentLeaf = nodeValues[index]
        currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
      print ("\n\nindex=", index, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
      currentNodeDecision = (GetNodeDecisions(currentDF, currentLeaf, index, className) )
      try: 
        next(tup for tup in nodeDecisions if tup[0] == index)
        print ("Node already added from brother leaf being null from hitting 'giniEndVal', i.e. leaf can't be improved anymore")
      except StopIteration: 
        nodeDecisions.append(currentNodeDecision )

    nodeDecisionsFileName =  nodeDecisionsFileName + ".csv"
    nodeDecisionsFile = open(nodeDecisionsFileName, 'w')
    nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
    for tup in nodeDecisions:
      nodeDecisionsFileCSV.writerow(tup)

  except KeyboardInterrupt:
    nodeValuesFileName = nodeValuesFileName + "_LastCompleteNode" + str(nodeCount-1) + ".csv"
    nodeValuesFile = open(nodeValuesFileName, 'w')
    nodeValuesFileCSV=csv.writer(nodeValuesFile)
    for tup in nodeValues:
      nodeValuesFileCSV.writerow(tup)
    nodeDFIDsFileName = nodeDFIDsFileName + "_LastCompleteNode" + str(nodeCount-1) + ".csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)

###################################################################
# With a fully expanded tree, get the decisions at each final
# node. Ouput will be a tuple = (Node number, the decision for the
# group that is less than the final cut, the decision for the 
# group greater than the last cut of the final node)
###################################################################
def GetNodeDecisions(df, leaf, index, className): 
      ltMaxCount = -100000000000000000000000000000000
      ltMaxClassVal = -100000000000000000000000000000000
      gtMaxCount = -100000000000000000000000000000000
      gtMaxClassVal = -100000000000000000000000000000000
      if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]):
        print ("END NODE")
        for classVal, row in  df[className].value_counts().to_dict().items():
          print ("\tclassVal=", classVal, "\trow=", int(row) )
          if int(row) > ltMaxCount:
            print ("\t\tltMaxCount=", ltMaxCount, "\tltMaxClassVal=", ltMaxClassVal)
            ltMaxCount = int(row)
            ltMaxClassVal = classVal
            print ("\t\tltMaxCount=", ltMaxCount, "\tltMaxClassVal=", ltMaxClassVal)
        return (index, ltMaxClassVal, np.NaN)
      
      print ("\tLESS THAN") #Getting the <= decision at node
      print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ][[className, leaf[2] ]]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ][[className, leaf[2] ]]) )
      for classVal, row in  df[ df[leaf[2]]<=leaf[3] ][className].value_counts().to_dict().items(): 
        print ("\tclassVal=", classVal, "\trow=", int(row) )
        if int(row) > ltMaxCount:
          print ("\t\tltMaxCount=", ltMaxCount, "\tltMaxClassVal=", ltMaxClassVal)
          ltMaxCount = int(row)
          ltMaxClassVal = classVal
          print ("\t\tltMaxCount=", ltMaxCount, "\tltMaxClassVal=", ltMaxClassVal)
      print("\tGREATER THAN") #Getting the <= decision at node
      for classVal, row in  df[ df[leaf[2]]>leaf[3] ][className].value_counts().to_dict().items(): 
        print ("\tclassVal=", classVal, "\trow=", int(row) )
        if int(row) > gtMaxCount:
          print ("\t\tgtMaxCount=", gtMaxCount, "\tgtMaxClassVal=", gtMaxClassVal)
          gtMaxCount = int(row)
          gtMaxClassVal = classVal
          print ("\t\tgtMaxCount=", gtMaxCount, "\tgtMaxClassVal=", gtMaxClassVal)
      return (index, ltMaxClassVal, gtMaxClassVal)

####################################################################
# Given a final Tree described by the nodes and their tple values
# described above and the decisions of those nodes,  make decisions
# of a set of points in a DF
####################################################################
def ClassifyWithTree(df_test, nodeDecisionsFileName, nodeValuesFileName, className, idColumn, outputFileName, maxDepth):
  print ("########################################################################\n Classifying test points with Tree from Make Tree\n########################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className] = np.nan 
  with open(nodeDecisionsFileName) as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    nodeDecisions = [tuple(line) for line in csv.reader(nodeDecisionsFile)]
  with open(nodeValuesFileName) as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    nodeValues = [tuple(line) for line in csv.reader(nodeValuesFile)]
  dfIDList = [ (0, df_test[idColumn].tolist()) ]
  maxNodeCount = 0
  for i in range(1,maxDepth+1): maxNodeCount += 2**i
  for ite in nodeValues:
    tup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]) )
    print ("\n\ttup=", tup )
    print ("\tdfIDList[tup[0][1]=", len(dfIDList[tup[0]][1])) 
    dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[tup[0]][1])]
    print ("\tdfCurr.head(1)\n", dfCurr.head(1) )
    if pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[2] == '' and pd.isnull(tup[1]): 
      print ("\tdf=", dfIDList[tup[0]][1])
      continue
    elif tup[2] == 'ThisIsAnEndNode' and pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[1] == 1.0:
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0])
      IDs = dfCurr[idColumn].tolist()
      df_Answers[ df_Answers[idColumn].isin(IDs) ] = decision[1]
      print ("\tdecision=", decision[1] )
      if tup[0] < maxNodeCount / 2:
        dfIDList.append( (tup[0]*2 + 1, [] ) )
        dfIDList.append( (tup[0]*2 + 2, [] ) )
    elif tup[0] < maxNodeCount / 2: 
      print ("\tlen(lt)=", len(dfCurr[ dfCurr[tup[2]] <= tup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[tup[2]] > tup[3] ]) )
      dfIDList.append( (tup[0]*2 + 1, dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist() ) ) 
      dfIDList.append( (tup[0]*2 + 2, dfCurr[ dfCurr[tup[2]] > tup[3] ][idColumn].tolist() ) )
    else:
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0])
      ltIDs = dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist()
      gtIDs = dfCurr[ dfCurr[tup[2]] >  tup[3] ][idColumn].tolist()
      df_Answers[ df_Answers[idColumn].isin(ltIDs) ] = decision[1]
      df_Answers[ df_Answers[idColumn].isin(gtIDs) ] = decision[2]
      print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
      del ltIDs, gtIDs, decision
    del dfCurr 
  #Writing the answers out
  df_Answers.to_csv(outputFileName + ".csv", sep=',', index=False)



def Boosting(df, algoritm, scoresAndWeightsFileName, nEstimators, algParamDict, nodeDecisionsFileName, nodeDFIDsFileName):
  currEst = 0
  dictNodesValues = {}
  weights = []
  while currEst < nEstimators:
    algParamDict['nodeDecisionsFileName'] = str(currEst) + "_" + algParamDict['nodeDecisionsFileName']
    algParamDict['nodeValuesFileName'] = str(currEst) + "_" + algParamDict['nodeValuesFileName']
    algParamDict['nodeDFIDsFileName'] = str(currEst) + "_" + algParamDict['nodeDFIDsFileName']
    MakeTree(df=df, **algParamDict) 
    score = (currEst, scoreTree(df=df, nodeDecisionsFileName=algParamDict['nodeDecisionsFileName'], nodeDFIDsFileName=algParamDict['nodeDFIDsFileName']) )
    





