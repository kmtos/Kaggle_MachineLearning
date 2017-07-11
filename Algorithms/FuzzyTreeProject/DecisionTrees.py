import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math
import random
from operator import itemgetter


#######################################################
# Given a df, this finds the column and value of
# The best split for the next two leaves based on gini
# tup = (node #, amount of gini increase, the column name, 
#        the value split at, range between splits)
#######################################################
def FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal, minSamplesSplit, df_weights):
  columns = [col for col in df if col != className and col != idColumn] # Get all columns but the class and ID column
  bestGiniSplit = (-1, -1, '', -100, -1)
  for classVal, rows in df[className].value_counts().to_dict().items(): # If the number of data points in the leaf are all of the same class, then the node ends
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
    bestSplitForCol = CalcBestGiniSplit(df, className, col, splits, minSamplesSplit, idColumn, df_weights) #Find the best split for this column
    if bestSplitForCol[0] > bestGiniSplit[1]: #See if this column provides the best Gini increase so far
      bestGiniSplit = (nodeCount, bestSplitForCol[0], col, bestSplitForCol[1], splitLength )
  if bestGiniSplit[1] < giniEndVal: 
    bestGiniSplit = (nodeCount, np.NaN, '', np.NaN, np.NaN) # Returns BlankNode if the best possible gini increase is less than the minimum for ending
    print ("LESS THAN giniEndVal")
  return bestGiniSplit
    
#############################################################
# given a column an it's splits, find the best gini increase
# tup = (amount of gini increase,the value split at)
#############################################################
def CalcBestGiniSplit(df, className, colName, splits, minSamplesSplit, idColumn, df_weights):
  uniqueClasses = df[className].unique()
  bestGiniIncrease = (-100, -1)
  counts = []
  for split in splits: # Make list of the groups of elements for each split to later calculate the best gini increase
    if len(df[df[colName]>split]) < minSamplesSplit or len(df[df[colName]<=split]) < minSamplesSplit: # Checks that leaf has at least the minimum # of data points
      continue
    GTIDs=df[ df[colName]>split][idColumn].tolist()
    LTIDs=df[ df[colName]<=split][idColumn].tolist()
    counts.append( ( df_weights.loc[(df_weights[idColumn].isin(GTIDs)) & (df_weights[className]==uniqueClasses[0])]["Weights"].sum(axis=0), # Now done with Weights
                     df_weights.loc[ df_weights[idColumn].isin(GTIDs)]["Weights"].sum(axis=0),  split,
                     df_weights.loc[(df_weights[idColumn].isin(LTIDs)) & (df_weights[className]==uniqueClasses[0])]["Weights"].sum(axis=0),  
                     df_weights.loc[ df_weights[idColumn].isin(LTIDs)]["Weights"].sum(axis=0) ) )
    # Below is the gini split without weights. Recently added with weights
    #counts.append( (len(df[ (df[className]==uniqueClasses[0]) & (df[colName]>split) ]), len(df[df[colName]>split]), split,
    #                len(df[ (df[className]==uniqueClasses[0]) & (df[colName]<=split) ]), len(df[df[colName]<=split])) )
  for tup in counts:  #Finds the best gini increase for the splits
    giniIncreaseGT = tup[0]/tup[1] * tup[0]/tup[1] +  (1 - tup[0]/tup[1]) * (1 - tup[0]/tup[1])
    giniIncreaseLT = tup[3]/tup[4] * tup[3]/tup[4] +  (1 - tup[3]/tup[4]) * (1 - tup[3]/tup[4])
    giniIncrease = giniIncreaseGT*tup[1]/(tup[1]+tup[4]) + giniIncreaseLT*tup[4]/(tup[1]+tup[4])
    if giniIncrease > bestGiniIncrease[0]: # If the current increase is bigger than the previously best increase, then reassign to new best gini increase
      bestGiniIncrease = (giniIncrease, tup[2])
  return bestGiniIncrease
  
##################################################################
# Makes a Decision Tree that saves output in three forms:
#    1) The tuple found with FindingBestSplit for each node
#    2) The node number and the DF ID's at that leaf
#    3) The Decision that was made at each leaf
##################################################################
def MakeTree(df, className, nGiniSplits, alpha, giniEndVal, maxDepth, idColumn, minSamplesSplit, df_weights, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName):
  print ("\n\n###################################\n Making a Decision Tree\n###################################")
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodeValues = [] #Node number,  amount of gini increase, the column name, the value split at, range between splits)
  nodeDFIds = [] #The point ID's that are in each leaf
  nodeCount = 0
  try:
    while nodeCount <= maxNodes: #checking that I haven't met the max node numb set by maxDepth
      if nodeCount == 0: #for the trunk or first tree node
        nodeValues.append(FindingBestSplit(df=df, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal, 
                                           minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) 
        nodeDFIds.append( (nodeCount, df[idColumn].tolist()) )
      else:
        parentTup = nodeValues[(nodeCount-1) // 2] #getting parent nodeValues tuple
        parentDFIDs = nodeDFIds[(nodeCount-1) // 2][1] #getting parent dataframe row ID's
        print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup, "\tlen(parentDFIDs)=", len(parentDFIDs))
        if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): # Make BlankNodes for leaves whose parents are End nodes or other BlankNodes
          nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
          nodeDFIds.append( (nodeCount, pd.DataFrame()) )
        else: # Create new node with the best gini increase and the df ID's and other important  information
          if nodeCount % 2  == 1: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] <= parentTup[3]) ] #Getting dataframe elements that are lower than the parent split
          else: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] > parentTup[3]) ] #getting dataframe elements that are greater than or equal to the parent split
          print ("len(dfCurr)=", len(dfCurr), "\tparentTup[3]=", parentTup[3] )
          nodeValues.append(FindingBestSplit(df=dfCurr, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal, 
                                             minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) # get next best split node Values
          nodeDFIds.append( (nodeCount, dfCurr[idColumn].tolist()) ) # Get next best split df ID's
          print ("######## NEW ########:", "nodeValues[nodeCount]=", nodeValues[nodeCount], "\tlen(nodeDFIds[1])=", len(nodeDFIds[1]))
          if not pd.isnull(nodeValues[nodeCount][3]): print ("len(lessThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] <= nodeValues[nodeCount][3]]), "\tlen(greaterThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] > nodeValues[nodeCount][3]]) ) 
      nodeCount += 1

    #Writing out the tuples for the nodes and cuts on which columns and dataframe ID's in each of the leaves
    print ("\n\n###########################\nWriting Out the Nodes, Values, and df ID's of the nodes\n###########################")
    nodeValuesFileName = nodeValuesFileName + ".csv"
    nodeValuesFile = open(nodeValuesFileName, 'w')
    nodeValuesFileCSV=csv.writer(nodeValuesFile)
    nodeValuesFileCSV.writerow(["NodeNumber,GiniIncrease,ColumnName,ValueOfSplit,RangeBetweenSplits"])
    for tup in nodeValues:
      nodeValuesFileCSV.writerow(tup)
    nodeDFIDsFileName = nodeDFIDsFileName + ".csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    nodeDFIdsFileCSV.writerow(["NodeNumber,ListOfID'sAtNode"])
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)
    
    #Getting the first non-dead leaf, i.e. leaf who's parent has a gini increase greater than the minimum for a leaf to end
    nodeDecisions = []
    minNodeNotLeaf = maxNodes
    for ite in range(maxNodes, maxNodes - 2**maxDepth, -1):
      index = ite 
      currentLeaf = nodeValues[index]
      currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
      while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]): # Check if node is a BlankNode and if so get it's parent until non-blank
        index = (ite-1) // 2
        currentLeaf = nodeValues[index]
        currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
      print ("\n\nindex=", index, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
      currentNodeDecision = (GetNodeDecisions(currentDF, currentLeaf, index, className) ) # Get the decision of the node based upon democracy
      try:  # This sees if the decision of a node is already added. From a sister BlankNode
        next(tup for tup in nodeDecisions if tup[0] == index)
        print ("Node already added from brother leaf being null from hitting 'giniEndVal', i.e. leaf can't be improved anymore")
      except StopIteration:  #If node is not found in nodeDecisions, then add it
        nodeDecisions.append(currentNodeDecision )

    #Write out the nodeDecisions
    nodeDecisionsFileName =  nodeDecisionsFileName + ".csv"
    nodeDecisionsFile = open(nodeDecisionsFileName, 'w')
    nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
    nodeDecisionsFileCSV.writerow(["NodeNumber,Decision_for_group_lessthan_cut_at_node,Decision_for_group_greaterthan_cut_at_node,nCorr_LT_group,nCorr_GT_group"])
    for tup in nodeDecisions:
      nodeDecisionsFileCSV.writerow(tup)

  except KeyboardInterrupt: # If you manually end "MakeTree" before it finishes, it writes out the nodes it has completed. Haven't written a function to use this yet, but will do in the future
    nodeValuesFileName = nodeValuesFileName + "_LastCompleteNode" + str(nodeCount-1) + ".csv"
    nodeValuesFile = open(nodeValuesFileName, 'w')
    nodeValuesFileCSV=csv.writer(nodeValuesFile)
    nodeValuesFileCSV.writerow(["NodeNumber,GiniIncrease,ColumnName,ValueOfSplit,RangeBetweenSplits"])
    for tup in nodeValues:
      nodeValuesFileCSV.writerow(tup)
    nodeDFIDsFileName = nodeDFIDsFileName + "_LastCompleteNode" + str(nodeCount-1) + ".csv"
    nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    nodeDFIdsFileCSV.writerow(["NodeNumber,ListOfID'sAtNode"])
    for tup in nodeDFIds:
      nodeDFIdsFileCSV.writerow(tup)

########################################################################
# With a fully expanded tree, get the decisions at each final
# node. Ouput will be a tuple = (Node number, the decision for the
# group that is less than the final cut, the decision for the 
# group greater than the last cut of the final node, number of 
# correctly labeled in lt group, number of correctly labeled in gt group
#########################################################################
def GetNodeDecisions(df, leaf, index, className): 
  ltMaxCount = -100000000000000000000000000000000
  ltMaxClassVal = -100000000000000000000000000000000
  gtMaxCount = -100000000000000000000000000000000
  gtMaxClassVal = -100000000000000000000000000000000
  if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): # See if this is a node where every element is the same class
    print ("END NODE")
    for classVal, row in  df[className].value_counts().to_dict().items():
      print ("\tclassVal=", classVal, "\trow=", int(row) )
      if int(row) > ltMaxCount:
        ltMaxCount = int(row)
        ltMaxClassVal = classVal
    print ("\t", (index, ltMaxClassVal, np.NaN, ltMaxCount, np.NaN) )
    return (index, ltMaxClassVal, np.NaN, ltMaxCount, np.NaN)
  
  print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
  print ("\tLESS THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[leaf[2]]<=leaf[3] ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the LT group of end node
    print ("\tclassVal=", classVal, "\trow=", int(row) )
    if int(row) > ltMaxCount:
      ltMaxCount = int(row)
      ltMaxClassVal = classVal
  print("\tGREATER THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[leaf[2]]>leaf[3] ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the GT group of end node
    print ("\tclassVal=", classVal, "\trow=", int(row) )
    if int(row) > gtMaxCount:
      gtMaxCount = int(row)
      gtMaxClassVal = classVal
  print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxCount, gtMaxCount))
  return (index, ltMaxClassVal, gtMaxClassVal, ltMaxCount, gtMaxCount)

####################################################################
# Given a final Tree described by the nodes and their tple values
# described above and the decisions of those nodes,  make decisions
# of a set of points in a DF
####################################################################
def ClassifyWithTree(df_test, className, idColumn, maxDepth, outputFileName, nodeDecisionsFileName, nodeValuesFileName):
  print ("\n\n########################################################################\n Classifying test points with Tree from Make Tree\n########################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className] = np.nan # Answer storage df
  print ("df_Answers=", df_Answers.head(10) )
  with open(nodeDecisionsFileName) as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName) as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]

  dfIDList = [ (0, df_test[idColumn].tolist()) ]
  maxNodeCount = 0
  for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes from maxDepth
  for ite in nodeValues: #Iterate through the nodes
    tup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]) ) # The File stores all info as strings. Cleaner to reassing type now instead of every instance.
    print ("\n\ttup=", tup )
    print ("\tdfIDList[tup[0][1]=", len(dfIDList[tup[0]][1])) 
    dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[tup[0]][1])] # Get the elements of df_test at node
    if pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[2] == '' and pd.isnull(tup[1]): # If decision of node from MakeTree is blank, then skip
      print ("\tdf=", dfIDList[tup[0]][1])
      continue
    elif tup[2] == 'ThisIsAnEndNode' and pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[1] == 1.0: # If decision of node from MakeTree is an EndNode, then proceed
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0]) # Get the decision of the End Node
      IDs = dfCurr[idColumn].tolist() # Get df_test elements that made it to this node following the tree structure
      df_Answers.loc[ df_Answers[idColumn].isin(IDs) , className] = decision[1] # Give the elements the appropriate class value from decision
      if tup[0] < maxNodeCount / 2: # If this EndNode isn't at the furthest depth, then add empty placeholders for future BlankNodes
        dfIDList.append( (tup[0]*2 + 1, [] ) )
        dfIDList.append( (tup[0]*2 + 2, [] ) )
    elif tup[0] < maxNodeCount / 2: # If element isn't an EndNode and also not at the furthest depth, then proceed
      print ("\tlen(lt)=", len(dfCurr[ dfCurr[tup[2]] <= tup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[tup[2]] > tup[3] ]) )
      dfIDList.append( (tup[0]*2 + 1, dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist() ) ) # Give the df_test ID's in the daughter LT leaf of current Node
      dfIDList.append( (tup[0]*2 + 2, dfCurr[ dfCurr[tup[2]] > tup[3] ][idColumn].tolist() ) )  # Give the df_test ID"s in the daughter Gt leaf of current Node
    else: # If not an EndNode, BlankNode, or a node NOT at the max depth, then get decisions there
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0]) # Get decision of Make Tree at node
      ltIDs = dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
      gtIDs = dfCurr[ dfCurr[tup[2]] >  tup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter GT leaf of current Node
      df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className] = decision[1] # Apply decision to LT group
      df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className] = decision[2] # Apply decision to GT group
      print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
      del ltIDs, gtIDs, decision # Delete containers to preserve memory, in case they don't already get deleted
    del dfCurr 
  #Writing the answers out
  print ("df_Answers=", df_Answers.head(10) )
  df_Answers.to_csv(outputFileName + ".csv", sep=',', index=False) #Write out the answers
