import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import string
import csv
import sys
import math
from collections import defaultdict
sys.path.insert(0, '/home/kyletos/Kaggle/Tools/')
from formatting import *

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)


df_train = pd.read_csv('training_data/train_step2.csv', header=0)

#id,qid1,qid2,question1,question2,is_duplicate,nWords1,nWords2,nChars1,nChars2,q1_stem,q2_stem,q1_stem_noStop,q2_stem_noStop,q1_stem_double,q2_stem_double,q1_stem_double_noStop,q2_stem_double_noStop,q1_stem_triple,q1_stem_triple_noStop,q2_stem_triple,q2_stem_triple_noStop

df_train = df_train.fillna(' ')

df_train['diffWords'] = df_train['nWords1'] - df_train['nWords2'] 
df_train['diffChars'] = df_train['nChars1'] - df_train['nChars2']

#Setting single, double, and triple stemmed words to lists for next step
df_train['q1_stem_list'] = df_train['q1_stem'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_list'] = df_train['q2_stem'].map(lambda x: x.split(" ")[0:] )
df_train['q1_stem_noStop_list'] = df_train['q1_stem_noStop'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_noStop_list'] = df_train['q2_stem_noStop'].map(lambda x: x.split(" ")[0:] )

df_train['q1_stem_double_list'] = df_train['q1_stem_double'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_double_list'] = df_train['q2_stem_double'].map(lambda x: x.split(" ")[0:] )
df_train['q1_stem_double_noStop_list'] = df_train['q1_stem_double_noStop'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_double_noStop_list'] = df_train['q2_stem_double_noStop'].map(lambda x: x.split(" ")[0:] )

df_train['q1_stem_triple_list'] = df_train['q1_stem_triple'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_triple_list'] = df_train['q2_stem_triple'].map(lambda x: x.split(" ")[0:] )
df_train['q1_stem_triple_noStop_list'] = df_train['q1_stem_triple_noStop'].map(lambda x: x.split(" ")[0:] )
df_train['q2_stem_triple_noStop_list'] = df_train['q2_stem_triple_noStop'].map(lambda x: x.split(" ")[0:] )

#Finding the number of matches between two questions
df_train['match_stem'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_list'], x['q2_stem_list']), axis=1 )
df_train['match_stem_noStop'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_noStop_list'], x['q2_stem_noStop_list']), axis=1 )
df_train['match_stem_double'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_double_list'], x['q2_stem_double_list']), axis=1 )
df_train['match_stem_double_noStop'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_double_noStop_list'], x['q2_stem_double_noStop_list']), axis=1 )
df_train['match_stem_triple'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_triple_list'], x['q2_stem_triple_list']), axis=1 )
df_train['match_stem_triple_noStop'] = df_train.apply(lambda x: twoStringListMatches(x['q1_stem_triple_noStop_list'], x['q2_stem_triple_noStop_list']), axis=1 )

df_train['nChars_match_stem'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_list'], x['q2_stem_list']), axis=1 )
df_train['nChars_match_stem_noStop'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_noStop_list'], x['q2_stem_noStop_list']), axis=1 )
df_train['nChars_match_stem_double'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_double_list'], x['q2_stem_double_list']), axis=1 )
df_train['nChars_match_stem_double_noStop'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_double_noStop_list'], x['q2_stem_double_noStop_list']), axis=1 )
df_train['nChars_match_stem_triple'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_triple_list'], x['q2_stem_triple_list']), axis=1 )
df_train['nChars_match_stem_triple_noStop'] = df_train.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_triple_noStop_list'], x['q2_stem_triple_noStop_list']), axis=1 )

df_train['sequential_match_stem'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_list'], x['q2_stem_list']), axis=1 )
df_train['sequential_match_stem_noStop'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_noStop_list'], x['q2_stem_noStop_list']), axis=1 )
df_train['sequential_match_stem_double'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_double_list'], x['q2_stem_double_list']), axis=1 )
df_train['sequential_match_stem_double_noStop'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_double_noStop_list'], x['q2_stem_double_noStop_list']), axis=1 )
df_train['sequential_match_stem_triple'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_triple_list'], x['q2_stem_triple_list']), axis=1 )
df_train['sequential_match_stem_triple_noStop'] = df_train.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_triple_noStop_list'], x['q2_stem_triple_noStop_list']), axis=1 )


"""
for row in df_train.itertuples():
  #if row.Index > 10: break
  #print ("\n", df_train.iloc[row.Index]['q1_stem'], "\n", df_train.iloc[row.Index]['q2_stem'], "\n", df_train.iloc[row.Index]['match_stem'] )
  #print ("\n", df_train.iloc[row.Index]['q1_stem_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_noStop'], "\n", df_train.iloc[row.Index]['match_stem_noStop'] )
  #print ("\n", df_train.iloc[row.Index]['q1_stem_double'], "\n", df_train.iloc[row.Index]['q2_stem_double'], "\n", df_train.iloc[row.Index]['match_stem_double'] )
  #print ("\n", df_train.iloc[row.Index]['q1_stem_double_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_double_noStop'], "\n", df_train.iloc[row.Index]['match_stem_double_noStop'] )
  #print ("\n", df_train.iloc[row.Index]['q1_stem_triple'], "\n", df_train.iloc[row.Index]['q2_stem_triple'], "\n", df_train.iloc[row.Index]['match_stem_triple'] )
  #print ("\n", df_train.iloc[row.Index]['q1_stem_triple_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_triple_noStop'], "\n", df_train.iloc[row.Index]['match_stem_triple_noStop'] )

  if row.Index < 5:
    print ("\n", df_train.iloc[row.Index]['q1_stem'], "\n", df_train.iloc[row.Index]['q2_stem'])
    #print ("\n", df_train.iloc[row.Index]['q1_stem_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_noStop'])
    #print ("\n", df_train.iloc[row.Index]['q1_stem_double'], "\n", df_train.iloc[row.Index]['q2_stem_double'])
    #print ("\n", df_train.iloc[row.Index]['q1_stem_double_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_double_noStop'])
    #print ("\n", df_train.iloc[row.Index]['q1_stem_triple'], "\n", df_train.iloc[row.Index]['q2_stem_triple'] )
    #print ("\n", df_train.iloc[row.Index]['q1_stem_triple_noStop'], "\n", df_train.iloc[row.Index]['q2_stem_triple_noStop'])
    print (numCharsOfStringListMatches( df_train.iloc[row.Index]['q1_stem_list'], df_train.iloc[row.Index]['q2_stem_list'] ) )
  if row.Index > 5: break
"""
#df_train.to_csv("training_data/train_step3.csv", index=False)
  
