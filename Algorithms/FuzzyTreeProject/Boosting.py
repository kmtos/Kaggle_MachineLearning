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
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/FuzzyTreeProject/')
from DecisionTrees import *

####################################################################
# This takes dict of parameters for MakeTree, the number of trees
# desired and makes that many of the normally produced files in 
# Make Trees. TreeErrors are also written based on tree correctness
####################################################################
def GetBoostingTreesErrorsAndWeights(df, nEstimators, algParamDict, colRandomness, rowRandomness, treeErrorFileName):
  if colRandomness > 1 or colRandomness < 0: print ("Give a colRandomness between 0-1 for the fraction of columns to be removed.") 
  if rowRandomness > 1 or rowRandomness < 0: print ("Give a rowRandomness between 0-1 for the fraction of row to be removed.") 
  currEst = nEstimators 
  treeError = []
  try:
    while currEst > 0: # Make a Tree for each of the desired estimators. Start at nEstimators and go down, so if run is stopped, then you can readjust the number to get the original # of Trees
      dfCurr = df[ random.sample(df.columns.tolist(), math.ceil(len(df.columns) * (1-colRandomness) ) ) ].copy() #Selecting a random portion of columns like a randomForest
      dfCurr = dfCurr.sample(math.ceil(len(dfCurr.index) * (1-rowRandomness) ) ) #Selecting random portion of rows for double randomness
      
      algParamDict['nodeDecisionsFileName'] = algParamDict['nodeDecisionsFileName'][:-1] + str(currEst) # Add currEst to Tree names to differentiate solutions to each estimator
      algParamDict['nodeValuesFileName'] = algParamDict['nodeValuesFileName'][:-1] + str(currEst)
      algParamDict['nodeDFIDsFileName'] = algParamDict['nodeDFIDsFileName'][:-1] + str(currEst)
      MakeTree(df=df, **algParamDict) 
      treeError.append( (currEst, GetTreeError(df=df, lenDFTotal=len(df.index), className=algParamDict['className'], df_weights=algParamDict['df_weights'], 
                                               nodeDecisionsFileName=algParamDict['nodeDecisionsFileName'], 
                                               nodeDFIdsFileName=algParamDict['nodeDFIDsFileName'], idColumn=algParamDict['idColumn'], nodeValuesFileName=algParamDict['nodeValuesFileName']) ) ) 
      algParamDict['df_weights'] = AlterWeights(df=df, df_weights=algParamDict['df_weights'], error=treeError[currEst][1], idColumn=algParamDict['idColumn'], className=algParamDict['className'], 
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
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeDFIdsFileName + ".csv") as nodeDFIdsFile:
    nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
    next(nodeDFIdsFileReader)
    nodeDFIds = [tuple(line) for line in nodeDFIdsFileReader]
  with open(nodeValuesFileName + ".csv") as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]

  totalCorrWeight = 0
  sumDFWeights = df_weights['Weights'].sum(axis=0) # Sum for denominator in total tree error
  for decisionTup in nodeDecisions: # iterate over the decisions to add the correctly identified one to "totalCorrWeight"
    dfIDs = next(iteTup[1] for iteTup in nodeDFIds if int(iteTup[0]) == int(decisionTup[0])) #Read in DFId's
    dfIDs = dfIDs.replace("[", "").replace("]", "").replace(",", "").split()
    dfIDs = [int(i) for i in dfIDs]
    nodeValuesTup = next(iteTup for iteTup in nodeValues if int(iteTup[0]) == int(decisionTup[0]) ) # Read in the nodeValues for cut value to get the groups of lt and gt ID's
    ltCorrIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == decisionTup[1])][idColumn].tolist()
    if pd.isnull(float(decisionTup[4]) ): # This part of the tuple is only null if every element in node has the same class. Then we ignore the gt, because I put everything in the lt group
      totalCorrWeight += df_weights.loc[df_weights[idColumn].isin(ltCorrIDs)].sum(axis=0) # Add correct weights to the total
    else: # Get the  ID's for the gt group where the ID's are in the parent, they are gt the current cut, and it has the same className value as the decision
      gtCorrIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == decisionTup[2])][idColumn].tolist() 
      totalCorrWeight += df_weights.loc[df_weights[idColumn].isin(ltCorrIDs)].sum(axis=0) + df_weights.loc[df_weights[idColumn].isin(gtCorrIDs)].sum(axis=0) # Add weights of above to total if correct
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
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName + ".csv") as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
  with open(nodeDFIdsFileName + ".csv") as nodeDFIdsFile:
    nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
    next(nodeDFIdsFileReader)
    nodeDFIds = [tuple(line) for line in nodeDFIdsFileReader]
  alpha = math.log1p((1 - error) / error) # exponent factor for adjustment of weights
  for decisionTup in nodeDecisions:
    dfIDs = next(iteTup[1] for iteTup in nodeDFIds if int(iteTup[0]) == decisionTup[0])
    dfIDs = dfIDs.replace("[", "").replace("]", "").replace(",", "").split()
    dfIDs = [int(i) for i in dfIDs]
    nodeValuesTup = next(iteTup for iteTup in nodeValues if int(iteTup[0]) == decisionTup[0])
    ltCorrIDs  = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == decisionTup[1])][idColumn].tolist() # Get correctly ID'd ID's in LT group
    ltWrongIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] != decisionTup[1])][idColumn].tolist() # Get incorrectly ID'd ID's in LT group
    df_weights['Weights'] = df_weights[df_weights[idColumn].isin(ltCorrIDs )]['Weights'] * math.exp(-1*alpha) # Make weights of correct ones less
    df_weights['Weights'] = df_weights[df_weights[idColumn].isin(ltWrongIDs)]['Weights'] * math.exp( 1*alpha) # Make weights of incorrect ones more
    if not pd.isnull(float(decisionTup[4]) ):
      gtCorrIDs  = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == decisionTup[2])][idColumn].tolist() # Get correctly Id'd Id's in GT group
      gtWrongIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] != decisionTup[2])][idColumn].tolist() # GEt incorrectly ID'd ID's in GT group
      df_weights['Weights'] = df_weights[df_weights[idColumn].isin(gtCorrIDs )]['Weights'] * math.exp(-1*alpha) # Make weights of correct ones less
      df_weights['Weights'] = df_weights[df_weights[idColumn].isin(gtWrongIDs)]['Weights'] * math.exp( 1*alpha) # Make weights of incorrect ones more
  nodeDFIdsFile.close()
  del nodeDFIdsFileReader
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  nodeValuesFile.close()
  del nodeValuesFileReader
  return df_weights

#############################################################
# For nEstimator trees, give the overall class decisions and
# the probability of point based on correctness. Should work 
# With >= 2 different classes, but haven't tested yet.
#############################################################
def CalssifyWithBoost(df_test, nEstimators, idColumn, className, treeErrorFileName, nodeValuesFileName, nodeDecisionsFileName, nodeDFIdsFileName, boostAnswersFileName):
  print ("\n\n########################################################################\n Classifying Boosted Tree\n######################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className + "_total"] = 0.0 # Total sum of the alphas over the nEstimator trees
  for classVal in df[className].unique(): 
    df_Answers[className + "_" + classVal] = 0.0 # Create new column for sum of alpha's for each unique className value
  with open(treeErrorFileName + ".csv") as treeErrorFile: 
    treeErrorFileReader = csv.reader(treeErrorFile)
    next(treeErrorFileReader)
    treeError = [tuple(line) for line in treeErrorFileReader]
  while currEst <= nEstimators: # Loop over the nEstimators number of trees in boost
    nodeValuesFileName =  nodeValuesFileName[:-1] + str(currEst)
    nodeDecisionsFileName =  nodeDecisionsFileName[:-1] + str(currEst)
    nodeDFIdsFileName = nodeDFIdsFileName[:-1] + str(currEst)
    currErrorTup = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == currEst)
    alpha = math.log1p((1 - error) / error) # exponent factor for weight of decision 
    with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
      nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
      next(nodeDecisionsFileReader)
      nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
    with open(nodeValuesFileName + ".csv") as nodeValuesFile:
      nodeValuesFileReader = csv.reader(nodeValuesFile)
      next(nodeValuesFileReader)
      nodeValues = [tuple(line) for line in nodeValuesFileReader]
    with open(nodeDFIdsFileName + ".csv") as nodeDFIdsFile:
      nodeDFIdsFileReader = csv.reader(nodeDFIdsFile)
      next(nodeDFIdsFileReader)


    dfIDList = [ (0, df_test[idColumn].tolist()) ] # List of node population of df_test based on the MakeTree nodeDecisions
    maxNodeCount = 0
    for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes based upon maxDepth
    for ite in nodeValues:
      nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]) ) # The File stores all info as strings. Cleaner to reassing type now instead of every instance.
      print ("\n\tnodeValueTup=", nodeValueTup )
      dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[nodeValueTup[0]][1])] #Get the population of df_test at current node
      if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): # Check if node is a BlankNode
        print ("\tdf=", dfIDList[nodeValueTup[0]][1])
        continue
      elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: #Check if node is an EndNode
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of the EndNode
        IDs = dfCurr[idColumn].tolist() #Get the elements of df_test at this node
        df_Answers[ df_Answers[idColumn].isin(IDs) ][className + "_" + decision[1]] += alpha #Apply the decision from the MakeTree to all df_test elements at present node
        df_answers[className + "_total"] += alpha
        print ("\tdecision=", decision[1], "\talpha=", alpha ) 
        if nodeValueTup[0] < maxNodeCount / 2: # If node is not at the maxDepth, then make blank placeholders for daughters in the iteration
          dfIDList.append( (nodeValueTup[0]*2 + 1, [] ) )
          dfIDList.append( (nodeValueTup[0]*2 + 2, [] ) )
      elif nodeValueTup[0] < maxNodeCount / 2: # If node is not a BlankNode or an EndNode and it is not at the max depth, then proceed
        print ("\tlen(lt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ]) )
        dfIDList.append( (nodeValueTup[0]*2 + 1, dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() ) ) # Add the LT group of df_test elements as daughter of node
        dfIDList.append( (nodeValueTup[0]*2 + 2, dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() ) )  # ADd the GT group of df_test elements as daughter of node
      else: # If node is not a BlankNode, EndNode, and is at the maxDepth of tree, then proceed
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of node for the LT and GT groups
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's that are LT of cut of final node
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's that are GT of cur of final node
        df_Answers[ df_Answers[idColumn].isin(ltIDs) ][className + "_" + decision[1]] += alpha # Apply the decisions of LT group from nodeDecisions to LT group of df_test elements
        df_Answers[ df_Answers[idColumn].isin(gtIDs) ][className + "_" + decision[2]] += alpha # Apply the decisions of GT group from nodeDecisions to GT group of df_test elements
        df_Answers[className + "_total"] += alpha
        print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
        del ltIDs, gtIDs, decision # Delete holders to preserve memory, if not already deleted by python
      del dfCurr 
     
    nodeDecisionsFile.close()
    del nodeDecisionsFileReader
    nodeValuesFile.close()
    del nodeValuesFileReader 

  df_Answers[className] = -1
  df_Answers[className + "_probability"] = -1
  for classVal in df[className].unique():
    df_Answers[className + "_" + classVal] = df_Answers[className + "_" + classVal] / df_Answers[className + "_total"] # Normalizing sums to total, to make a probability
    df_Answers[df_Answers[className + "_" + classVal] > df_Answers[className]] = classVal # if current classVal prob is greater, reassign answer to the classVal
    df_Answers[df_Answers[className] == classVal][className + "_probability"] = df_Answers[className + "_" + classVal] # If current classVal prob got changed, reassign probab
  df_Answers.to_csv(boostAnswersFileName + ".csv", sep=',', index=False) #Write out the answers



