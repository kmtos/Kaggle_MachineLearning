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
from DecisionTrees import *
from FuzzyTrees_Classification import *
from Boosting import *
#################################################
# Update class decision Score by taking percent
# of memberhship and put it all towards the 
# decision at the node it currently is a part of
#################################################
def GetCurrentNodeFuzziedWeight(memberships, nodeNumber):
  nodeTup = [tup for tup in memberships if int(tup[0]) == int(nodeNumber) ] # Find the percent of membership in this node 
  return nodeTup[0][1]  # Add the membership percent to the decision of this node dictated in calling the function

def ReturnNodePointsToParent(memberships, nodeNumber):
  nodeTup = [tup for tup in memberships if int(tup[0]) == int(nodeNumber)] # Find the percent of membership in this node
  memberships.remove( nodeTup[0])  
  parentNodeNum = (nodeTup[0][0]-1)//2
  parentTup = [tup for tup in memberships if int(tup[0]) == int(parentNodeNum)]
  new = []
  if parentTup:
    memberships.remove(parentTup[0])
    new.append( (parentNodeNum, nodeTup[0][1] + parentTup[0][1]) )
    return new + memberships 
  new.append( (parentNodeNum, nodeTup[0][1]) )
  return new + memberships

##################################################################
# Makes a Decision Tree that saves output in three forms:
#    1) The tuple found with FindingBestSplit for each node
#    3) The Decision that was made at each leaf
##################################################################
def MakeFuzzyTree(df, className, nGiniSplits, giniEndVal, maxDepth, idColumn, minSamplesSplit, duality, df_weights, nodeValuesFileName, nodeDecisionsFileName):
  print ("\n\n###################################\n Making a Decision Tree\n###################################")
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodeValues = [] #Node number,  amount of gini increase, the column name, the value split at, range between splits)
  nodeCount = 0
  df['Memberships'] = df_weights.apply(lambda row: [ (0, row['Weights']) ], axis=1) # initialize the Memberships of the nodes, with weight of each point
  print ("unique Weights=", df_weights['Weights'].unique() )  
  df['MembershipNodeList'] = [ [0] for _ in range( len(df)) ] # Initialize all points to have their list of nodes they are a part of be only node 0
  while nodeCount <= maxNodes: #checking that I haven't met the max node numb set by maxDepth
    if nodeCount == 0: #for the trunk or first tree node
      nodeValues.append(FindingBestSplit(df=df, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal,  #If first node find split
                                         minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) 
      df['Memberships'] = df.apply(lambda row:
                                    FuzzyMembershipLinear( value=row[nodeValues[0][2]], split=nodeValues[0][3], splitLength=nodeValues[0][4], duality=duality,
                                    previousList=row['Memberships'], nodeNumber=0, daughterEndNode=None ), axis=1 ) # Sets points membership via linear range
      df['MembershipNodeList'] = df.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) # Updates Membership's Node list depending on the points membership
    else:
      parentTup = nodeValues[(nodeCount-1) // 2] #getting parent nodeValues tuple
      print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup)
      if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): # Make BlankNodes for leaves whose parents are End nodes or other BlankNodes
        nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
      else: # Create new node with the best gini increase and the df ID's and other important  information
        print ("nodeCount=", nodeCount)
        df_Curr = df[ df['MembershipNodeList'].apply(lambda x: True if nodeCount in x else False) ].copy() # Get the nodes that have a membership in current node number
        print ("len(df_Curr.index)=", len(df_Curr.index))
        if len(df_Curr.index) < minSamplesSplit:
          print ("Too Few to be able to split")
          nodeValues.append( (nodeCount, np.NaN, '', np.NaN, np.NaN) )
        else: 
          nodeValues.append(FindingBestSplit(df=df_Curr, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal, 
                                           minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) # get next best split node Values
          if not pd.isnull(nodeValues[nodeCount][4]) and not pd.isnull(nodeValues[nodeCount][3]) and nodeCount < maxNodes/2: 
            df_Curr['Memberships'] = df_Curr.apply(lambda row: FuzzyMembershipLinear( value=row[nodeValues[nodeCount][2]], split=nodeValues[nodeCount][3], splitLength=nodeValues[nodeCount][4], 
                                             duality=duality, previousList=row['Memberships'], nodeNumber=nodeCount, daughterEndNode=None ), axis=1 ) # Sends points membership via linear range
            df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) # Updates Membership's Node list depending on the points membership
            IDs = df_Curr[idColumn].tolist() # Get the df ID's in the
            df.loc[ df[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships'] # Update df 'Membership' with df_Curr's values
            df.loc[ df[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] #Update df 'MembershipNodeList' with df_Curr's values
          elif nodeValues[nodeCount][2] == '':
            df_Curr['Memberships'] = df_Curr.apply(lambda row: ReturnNodePointsToParent(memberships=row['Memberships'], nodeNumber=nodeCount), axis=1)
            df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1)
            IDs = df_Curr[idColumn].tolist() # Get the df ID's in the
            df.loc[ df[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships'] # Update df 'Membership' with df_Curr's values
            df.loc[ df[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] #Update df 'MembershipNodeList' with df_Curr's values
             
          print ("######## NEW ########:", "nodeValues[nodeCount]=", nodeValues[nodeCount], "\tlen(nodeDFIds[", nodeCount, "][1])=", 
                 len( df[ df['MembershipNodeList'].apply(lambda x: True if nodeCount in x else False)].index ) )
          if not pd.isnull(nodeValues[nodeCount][3]): 
            print ("len(lessThan)=", len(df_Curr.loc[df_Curr[nodeValues[nodeCount][2]] <= nodeValues[nodeCount][3]]), "\tlen(greaterThan)=", 
                   len(df_Curr.loc[df_Curr[nodeValues[nodeCount][2]] > nodeValues[nodeCount][3]]) ) 
    nodeCount += 1

  #Writing out the tuples for the nodes and cuts on which columns and dataframe ID's in each of the leaves
  print ("\n\n###########################\nWriting Out the Nodes, Values, and df ID's of the nodes\n###########################")
  nodeValuesFileName = nodeValuesFileName + ".csv"
  nodeValuesFile = open(nodeValuesFileName, 'w')
  nodeValuesFileCSV=csv.writer(nodeValuesFile)
  nodeValuesFileCSV.writerow(["NodeNumber,GiniIncrease,ColumnName,ValueOfSplit,RangeBetweenSplits"])
  for tup in nodeValues:
    nodeValuesFileCSV.writerow(tup)
  
  #Getting the first non-dead leaf, i.e. leaf who's parent has a gini increase greater than the minimum for a leaf to end
  nodeDecisions = []
  minNodeNotLeaf = maxNodes
  print ("df_weights['Weights'].sum(axis=0)=", df_weights['Weights'].sum(axis=0) )
  for ite in range(maxNodes, maxNodes - 2**maxDepth, -1):
    index = ite 
    print ("\n\nindex=", index)
    currentLeaf = nodeValues[index]
    currentDF = df[ df['MembershipNodeList'].apply(lambda x: True if index in x else False) ].copy() # Get the points in the current node
    currentDF_weights = df_weights.loc[ df[idColumn].isin(currentDF[idColumn].tolist() )]
    gt_or_lt = 0 # Use this to see if the sister node is GT or LT of the split of the current node
    soloNode = False # Check for if the sister node does NOT have a decision
    while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]): # Check if node is a BlankNode and if so get it's parent until non-blank
      sisterNode = index-1 if index % 2 == 0 else index+1
      gt_or_lt = 1 if index % 2 == 0 else -1
      soloNode = False 
      if  pd.isnull(nodeValues[sisterNode][1]) or nodeValues[sisterNode][2] != '' or pd.isnull(nodeValues[sisterNode][3]) or pd.isnull(nodeValues[sisterNode][4]): soloNode = True
      index = (index-1) // 2
      currentLeaf = nodeValues[index]
      currentDF = df[ df['MembershipNodeList'].apply(lambda x: True if index in x else False) ].copy()
      currentDF_weights = df_weights.loc[ df[idColumn].isin(currentDF[idColumn].tolist() )]
    print ("index=", index, "\tgt_or_lt=", gt_or_lt, "\tsoloNode=", soloNode, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
    currentDF['CurrNodeFuzziedWeight'] = currentDF.apply( lambda row: GetCurrentNodeFuzziedWeight( memberships=row['Memberships'], nodeNumber=index), axis=1)
    print ("currentDF[[className, 'CurrNodeFuzziedWeight', 'Memberships']]=\n", currentDF[[className, 'CurrNodeFuzziedWeight', 'Memberships']] )
    currentNodeDecision = (GetFuzzyWeightedNodeDecisions(df=currentDF, leaf=currentLeaf, index=index, className=className, soloNodeDecision=soloNode, gt_or_lt=gt_or_lt, idColumn=idColumn) ) # Get the decision of the node
    try:  # This sees if the decision of a node is already added. From a sister BlankNode
      next (tup for tup in nodeDecisions if tup[0] == index)
      print ("Decision already included from other daugther node")
    except StopIteration:  #If node is not found in nodeDecisions, then add it
      nodeDecisions.append(currentNodeDecision )

  #Write out the nodeDecisions
  nodeDecisionsFileName =  nodeDecisionsFileName + ".csv"
  nodeDecisionsFile = open(nodeDecisionsFileName, 'w')
  nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
  nodeDecisionsFileCSV.writerow(["NodeNumber,LT_className_decision,GT_className_decision,LT_WeightCorrect,GT_WeightCorrect,LT_TotalWeight,GT_TotalWeight"])
  for tup in nodeDecisions:
    nodeDecisionsFileCSV.writerow(tup)

#########################################################################
# With a Finished Tree, get the decision for the LT and GT group at each 
# node bsed upon a weighted democracy. Tuple=(node #, LT decision, GT 
# decision, LT Weight correct, GT weight correct, LT total weight, GT
# total Weight.)
#########################################################################
def GetFuzzyWeightedNodeDecisions(df, leaf, index, className, soloNodeDecision, gt_or_lt, idColumn):
  ltMaxWeight = -100000000000000000000000000000000
  ltMaxClassVal = -100000000000000000000000000000000
  gtMaxWeight = -100000000000000000000000000000000
  gtMaxClassVal = -100000000000000000000000000000000
  ltTotalWeight = -1
  gtTotalWeight = -1
  if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): # See if this is a node where every element is the same class
    print("This is An End Node")
    return (index, df[className].unique()[0], np.NaN, df['CurrNodeFuzziedWeight'].sum(axis=0), np.NaN, 
            df['CurrNodeFuzziedWeight'].sum(axis=0), np.NaN)
  
  if (soloNodeDecision):
    df_IDs = df[ df[leaf[2]]> leaf[3] ][idColumn].tolist() if gt_or_lt > 0 else df[ df[leaf[2]]<=leaf[3] ][idColumn].tolist()      
    totalWeight = df['CurrNodeFuzziedWeight'].sum(axis=0)
    for classVal, row in  df[ df[idColumn].isin(df_IDs)  ][className].value_counts().to_dict().items():
      currWeight = df[df[className] == classVal ]['CurrNodeFuzziedWeight'].sum(axis=0)
      print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > ltMaxWeight:
        ltMaxWeight = currWeight
        ltMaxClassVal = classVal
    if gt_or_lt > 0: 
      print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. GT Node Decision.")
      return (index, np.NaN, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight)
    else:            
      print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. LT Node Decision.")
      return (index, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight, np.NaN)

  print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
  print ("\tLESS THAN") #Getting the <= decision at node
  ltTotalWeight = df[ df[leaf[2]]<=leaf[3] ]['CurrNodeFuzziedWeight'].sum(axis=0)
  gtTotalWeight = df[ df[leaf[2]]> leaf[3] ]['CurrNodeFuzziedWeight'].sum(axis=0)
  for classVal, row in  df[ df[leaf[2]]<=leaf[3]  ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the LT group of end node
    currWeight = df[ (df[leaf[2]]<=leaf[3]) & (df[className] == classVal) ]['CurrNodeFuzziedWeight'].sum(axis=0)
    print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
    if currWeight > ltMaxWeight:
      ltMaxWeight = currWeight
      ltMaxClassVal = classVal
  print("\tGREATER THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[leaf[2]]> leaf[3] ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the GT group of end node
    currWeight = df[ (df[leaf[2]]> leaf[3]) & (df[className] == classVal) ]['CurrNodeFuzziedWeight'].sum(axis=0)
    print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
    if currWeight > gtMaxWeight:
      gtMaxWeight = currWeight
      gtMaxClassVal = classVal
  print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight) )
  return (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight)

####################################################################
# This takes dict of parameters for MakeTree, the number of trees
# desired and makes that many of the normally produced files in 
# Make Trees. TreeErrors are also written based on tree correctness
####################################################################
def GetFuzzyBoostingTreesErrorsAndWeights(df, nEstimators, rateOfChange, df_weights, paramDict, colRandomness, rowRandomness, treeErrorFileName, middleStart=False, middleStartNum=-1):
  if colRandomness > 1 or colRandomness < 0: print ("Give a colRandomness between 0-1 for the fraction of columns to be removed."
)
  if rowRandomness > 1 or rowRandomness < 0: print ("Give a rowRandomness between 0-1 for the fraction of row to be removed.")
  if middleStart == False or middleStartNum == -1: currEst = nEstimators
  else:
    currEst = middleStartNum
    df_weights = pd.read_csv("Answers/DF_WEIGHTS.csv")
  treeError = []
  try:
    while currEst > 0: # Make a Tree for each of the desired estimators. Start at nEstimators and go down, so if run is stopped, then you can readjust the number to get the original # of Trees
      columnsList = [ ite for ite in df.columns.tolist() if ite != paramDict['className'] and ite != paramDict['idColumn'] ]
      columnsList.extend((paramDict['className'], paramDict['idColumn']) )
      dfCurr = df[ random.sample(columnsList, math.ceil(len(columnsList) * (1-colRandomness) ) ) ].copy() #Selecting a random potion of columns like a randomForest
      dfCurr[paramDict['className'] ] = df[paramDict['className']]
      dfCurr[paramDict['idColumn'] ] = df[paramDict['idColumn']]
      dfCurr = dfCurr.sample(math.ceil(len(dfCurr.index) * (1-rowRandomness) ) ) #Selecting random portion of rows for double randomness
      dfCurr_weights = df_weights[ df_weights[paramDict['idColumn']].isin(dfCurr[paramDict['idColumn']].tolist() )].copy()
      #dfCurr_NotIn = df[ ~dfCurr[idColumn].isin(dfCurr[idColumn].tolist()) ].copy()
      print (paramDict['nodeDecisionsFileName'].rstrip('1234567890') )
      paramDict['nodeDecisionsFileName'] = paramDict['nodeDecisionsFileName'].rstrip('1234567890') + str(currEst) # Add currEst to Tree names to differentiate solutions to each estimator
      paramDict['nodeValuesFileName'] = paramDict['nodeValuesFileName'].rstrip('1234567890') + str(currEst)
      print ("#############################################\n#############################################\n  STARTING ESTIMATOR", currEst, "\n#############################################\n#############################################")
      TEMP = MakeFuzzyTree(df=dfCurr, df_weights=dfCurr_weights, **paramDict) # Make the Tree for currEst
      if TEMP == "ERROR": raise UnboundLocalError("\n\nRunning Ended Early")
      treeError.append( (currEst, GetTreeError(df=dfCurr, className=paramDict['className'], df_weights=dfCurr_weights, #Getting error for MakeTree currEst
                                               idColumn=paramDict['idColumn'], nodeDecisionsFileName=paramDict['nodeDecisionsFileName'] ) ) )
      dfCurr_weights = AlterWeights(df=dfCurr, df_weights=dfCurr_weights, error=next(i[1] for i in treeError if i[0] == currEst), rateOfChange=rateOfChange, idColumn=paramDict['idColumn'],
                                    className=paramDict['className'], nodeDecisionsFileName=paramDict['nodeDecisionsFileName'], nodeValuesFileName=paramDict['nodeValuesFileName'])
      print ("BEFORE: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
      IDs = dfCurr_weights[paramDict['idColumn']].tolist()  # Getting list of IDs of data points included from rowRandomness
      df_weights.loc[ df_weights[paramDict['idColumn']].isin( IDs ), 'Weights'] = dfCurr_weights['Weights'] # Changing the official weights of those points included fromt he rowRandomness
      print ("After: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
      for ite in df_weights['Weights'].unique():
        print ("\tlen('Weights' ==", ite, ")=", len(df_weights[ df_weights['Weights'] == ite].index) )
      df_weights['Weights'] = df_weights['Weights'] / df_weights['Weights'].sum(axis=0) # Normalizing weights to 1 after changing them
      print ("\n\n###############################################################\nFINAL SCORE FOR TREE #", currEst, "is 1-error=", 1-next(i[1] for i in treeError if i[0] == currEst),"\n###############################################################")
      currEst -= 1

    # Writing the TreeErrors
    treeErrorFileName =  treeErrorFileName + ".csv"
    if os.path.isfile(treeErrorFileName): treeErrorFile = open(treeErrorFileName, 'a')
    else: treeErrorFile = open(treeErrorFileName, 'w')
    treeErrorFileCSV=csv.writer(treeErrorFile)
    treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
    for tup in treeError:
      treeErrorFileCSV.writerow(tup)

  except (KeyboardInterrupt,UnboundLocalError): #If user stops runnig, still save 
      treeErrorFileName =  treeErrorFileName + "_Incomplete.csv"
      if os.path.isfile(treeErrorFileName):
        treeErrorFile = open(treeErrorFileName, 'a')
        treeErrorFileCSV=csv.writer(treeErrorFile)
      else:
        treeErrorFile = open(treeErrorFileName, 'w')
        treeErrorFileCSV=csv.writer(treeErrorFile)
        treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
      for tup in treeError:
        treeErrorFileCSV.writerow(tup)
      df_weights.to_csv("Answers/DF_WEIGHTS.csv", sep=',', index=False)



