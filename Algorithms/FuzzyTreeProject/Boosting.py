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


####################################################################
# This takes dict of parameters for MakeTree, the number of trees
# desired and makes that many of the normally produced files in 
# Make Trees. TreeErrors are also written based on tree correctness
####################################################################
def GetBoostingTreesErrorsAndWeights(df, algoritm, treeErrorFileName, nEstimators, algParamDict, df_weights, colRandomness=0, rowRandomness=0):
  if colRandomness > 1.0 or colRandomness < 0.0: print ("Give a colRandomness between 0-1 for the fraction of columns to be removed.") 
  if rowRandomness > 1.0 or rowRandomness < 0.0: print ("Give a rowRandomness between 0-1 for the fraction of row to be removed.") 
  currEst = nEstimators 
  treeError = []
  try:
    while currEst > 0: # Make a Tree for each of the desired estimators. Start at nEstimators and go down, so if run is stopped, then you can readjust the number to get the original # of Trees
      dfCurr = df[[ random.sample(df.columns, math.ceil(len(df.columns) * (1-colRandomness) ) ) ]].copy() #Selecting a random portion of columns like a randomForest
      dfCurr = dfCurr.iloc[[ random.sample(dfCurr[algParamDict['idColumn']], math.ceil(len(dfCurr.index) * (1-rowRandomness) ) ) ]]#Selecting random portion of rows for double randomness
      algParamDict['nodeDecisionsFileName'] = str(currEst) + "_" + algParamDict['nodeDecisionsFileName'] # Add currEst to Tree names to differentiate solutions to each estimator
      algParamDict['nodeValuesFileName'] = str(currEst) + "_" + algParamDict['nodeValuesFileName']
      algParamDict['nodeDFIDsFileName'] = str(currEst) + "_" + algParamDict['nodeDFIDsFileName']
      MakeTree(df=df, df_weights=df_weights, **algParamDict) 
      treeError.append( (currEst, GetTreeError(df=df, lenDFTotal=len(df.index), className=algParamDict['className'], df_weight=df_weights, nodeDecisionsFileName=algParamDict['nodeDecisionsFileName'], 
                                               nodeDFIdsFileName=algParamDict['nodeDFIDsFileName'], idColumn=algParamDict['idColumn'], nodeValuesFileName=algParamDict['nodeValuesFileName']) ) ) 
      df_weights = AlterWeights(df=df, df_weights=df_weights, error=treeError[currEst][1], idColumn=algParamDict['idColumn'], className=algParamDict['className'], 
                                nodeDecisionsFileName=algParamDict['nodeDecisionsFileName'], nodeDFIDsFileName=algParamDict['nodeDFIDsFileName'] ,nodeValuesFileName=algParamDict['nodeValuesFileName'])
      print ("\n\n###############################################################\nFINAL SCORE FOR TREE #", currEst, "is", treeError[currEst][1],"\n###############################################################")
      currEst -= 1 
  
      # Writing the TreeErrors
      treeErrorFileName =  treeErrorFileName + ".csv"
      treeErrorFile = open(treeErrorFileName, 'w')
      treeErrorFileCSV=csv.writer(treeErrorFile)
      treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
      for tup in treeError:
        treeErrorFileCSV.writerow(tup)

  except KeyboardInterrupt: #If user stops runnig, still save 
      treeErrorFileName =  treeErrorFileName + "_LastCompletedEst_" + currEst + ".csv"
      treeErrorFile = open(treeErrorFileName, 'w')
      treeErrorFileCSV=csv.writer(treeErrorFile)
      treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
      for tup in treeError:
        treeErrorFileCSV.writerow(tup)

########################################################
# Takes Set of NodeDecisions and uses the nCorr to get
# the weight of the correctness
########################################################
def GetTreeError(df, lenDFTotal, className, df_weights, idColumn, nodeValuesFileName, nodeDecisionsFileName, nodeDFIdsFileName):
  with open(nodeDecisionsFileName) as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeDFIdsFileName) as nodeDFIdsFile:
    nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
    next(nodeDFIdsFileReader)
    nodeDFIds = [tuple(line) for line in nodeDFIdsFileReader]
  with open(nodeValuesFileName) as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]

  totalCorrWeight = 0
  sumDFWeights = df_weights['Weights'].sum(axis=0)
  for decisionTuple in nodeDecisions: 
    dfIDs = next(iteTup for iteTup in nodeDFIds if int(iteTup[0]) == decisiontup[0])
    nodeValuesTup = next(iteTup for iteTup in nodeValues if int(iteTup[0]) == decisiontup[0])
    ltCorrIDs = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == decisionTuple[1])][idColumn].tolist()
    if pd.isnull(float(decisionTuple[4]) ): 
      totalCorrWeight += df_weights.loc[df_weights[idColumn].isin(ltCorrIDs)].sum(axis=0)
    else: 
      gtCorrIDs = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == decisionTuple[2])][idColumn].tolist()
      totalCorrWeight += df_weights.loc[df_weights[idColumn].isin(ltCorrIDs)].sum(axis=0) + df_weights.loc[df_weights[idColumn].isin(gtCorrIDs)].sum(axis=0)
  nodeDFIdsFile.close()
  del nodeDFIdsFileReader
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  nodeValuesFile.close()
  del nodeValuesFileReader
  return float(totalCorrWeight / sumDFWeights)


###########################################################
# Given the df, and the nodeDecisions, node DF IDs and 
# the nodeValues, alter the weights based upon correctness
###########################################################
def AlterWeights(df, error, idColumn, className, nodeDecisionsFileName, nodeDFIDsFileName, nodeValuesFileName):
  with open(nodeDecisionsFileName) as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName) as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
  with open(nodeDFIdsFileName) as nodeDFIdsFile:
    nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
    next(nodeDFIdsFileReader)
    nodeDFIds = [tuple(line) for line in nodeDFIdsFileReader]
  alpha = math.log1p((1 - error) / error)
  for decisionTup in nodeDecisions:
    dfIDs = next(iteTup for iteTup in nodeDFIds if int(iteTup[0]) == decisiontup[0])
    nodeValuesTup = next(iteTup for iteTup in nodeValues if int(iteTup[0]) == decisiontup[0])
    ltCorrIDs  = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == decisionTuple[1])][idColumn].tolist()
    ltWrongIDs = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] != decisionTuple[1])][idColumn].tolist()
    df_weights['Weights'] = df_weights[df_weights[idColumn].isin(ltCorrIDs )]['Weights'] * math.exp(-1*alpha)
    df_weights['Weights'] = df_weights[df_weights[idColumn].isin(ltWrongIDs)]['Weights'] * math.exp( 1*alpha)
    if not pd.isnull(float(decisionTuple[4]) ):
      gtCorrIDs  = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == decisionTuple[2])][idColumn].tolist()
      gtWrongIDs = df[ (df[idColumn].isin(parentDFIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] != decisionTuple[2])][idColumn].tolist()
      df_weights['Weights'] = df_weights[df_weights[idColumn].isin(gtCorrIDs )]['Weights'] * math.exp(-1*alpha)
      df_weights['Weights'] = df_weights[df_weights[idColumn].isin(gtWrongIDs)]['Weights'] * math.exp( 1*alpha)
  nodeDFIdsFile.close()
  del nodeDFIdsFileReader
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  nodeValuesFile.close()
  del nodeValuesFileReader
  return df_weights


def CalssifyWithBoost(df_test, treeErrorFileName, nodeValuesFileName, nEstimators, nodeDecisionsFileName, nodeDFIdsFileName, idColumn, className):
  print ("\n\n########################################################################\n Classifying Boosted Tree\n######################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className + "Sum"] = 0.0
  while currEst <= nEstimators:
    nodeValuesFileName = str(currEst) + "_" + nodeValuesFileName
    nodeDecisionsFileName = str(currEst) + "_" + nodeDecisionsFileName
    nodeDFIdsFileName = str(currEst) + "_" + nodeDFIdsFileName
    with open(nodeDecisionsFileName) as nodeDecisionsFile:
      nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
      next(nodeDecisionsFileReader)
      nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
    with open(nodeValuesFileName) as nodeValuesFile:
      nodeValuesFileReader = csv.reader(nodeValuesFile)
      next(nodeValuesFileReader)
      nodeValues = [tuple(line) for line in nodeValuesFileReader]
    with open(nodeDFIdsFileName) as nodeDFIdsFile:
      nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
      next(nodeDFIdsFileReader)
      nodeDFIds = [tuple(line) for line in nodeDFIdsFileReader]

    dfIDList = [ (0, df_test[idColumn].tolist()) ] # List of node population of df_test based on the MakeTree nodeDecisions
    maxNodeCount = 0
    for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes based upon maxDepth
    for ite in nodeValues:
      tup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]) ) # The File stores all info as strings. Cleaner to reassing type now instead of every instance.
      print ("\n\ttup=", tup )
      dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[tup[0]][1])] #Get the population of df_test at current node
      if pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[2] == '' and pd.isnull(tup[1]): # Check if node is a BlankNode
        print ("\tdf=", dfIDList[tup[0]][1])
        continue
      elif tup[2] == 'ThisIsAnEndNode' and pd.isnull(tup[3]) and pd.isnull(tup[4]) and tup[1] == 1.0: #Check if node is an EndNode
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0]) # Get decision of the EndNode
        IDs = dfCurr[idColumn].tolist() #Get the elements of df_test at this node
        df_Answers[ df_Answers[idColumn].isin(IDs) ] = decision[1] #Apply the decision from the MakeTree to all df_test elements at present node
        print ("\tdecision=", decision[1] ) 
        if tup[0] < maxNodeCount / 2: # If node is not at the maxDepth, then make blank placeholders for daughters in the iteration
          dfIDList.append( (tup[0]*2 + 1, [] ) )
          dfIDList.append( (tup[0]*2 + 2, [] ) )
      elif tup[0] < maxNodeCount / 2: # If node is not a BlankNode or an EndNode and it is not at the max depth, then proceed
        print ("\tlen(lt)=", len(dfCurr[ dfCurr[tup[2]] <= tup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[tup[2]] > tup[3] ]) )
        dfIDList.append( (tup[0]*2 + 1, dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist() ) ) # Add the LT group of df_test elements as daughter of node
        dfIDList.append( (tup[0]*2 + 2, dfCurr[ dfCurr[tup[2]] > tup[3] ][idColumn].tolist() ) )  # ADd the GT group of df_test elements as daughter of node
      else: # If node is not a BlankNode, EndNode, and is at the maxDepth of tree, then proceed
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == tup[0]) # Get decision of node for the LT and GT groups
        ltIDs = dfCurr[ dfCurr[tup[2]] <= tup[3] ][idColumn].tolist() # Get the df_test ID's that are LT of cut of final node
        gtIDs = dfCurr[ dfCurr[tup[2]] >  tup[3] ][idColumn].tolist() # Get the df_test ID's that are GT of cur of final node
        df_AnswersTEMP[ df_Answers[idColumn].isin(ltIDs) ] = decision[1] # Apply the decisions of LT group from nodeDecisions to LT group of df_test elements
        df_AnswersTEMP[ df_Answers[idColumn].isin(gtIDs) ] = decision[2] # Apply the decisions of GT group from nodeDecisions to GT group of df_test elements
        print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
        del ltIDs, gtIDs, decision # Delete holders to preserve memory, if not already deleted by python
      del dfCurr 
      
  
    nodeDecisionsFile.close()
    del nodeDecisionsFileReader
    nodeValuesFile.close()
    del nodeValuesFileReader 
