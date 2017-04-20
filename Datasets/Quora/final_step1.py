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
import pickle
pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)


df_test = pd.read_csv('testing_data/test_singles_doubles_triples.csv', header=0)
#df_test = pd.read_pickle('testing_data/test_singles.p')
try:
  """
  df_test['nWords1'] = df_test['question1'].map(lambda x: len(x.split(' ') ) if isinstance(x, str) else 0 ) 
  df_test['nWords2'] = df_test['question2'].map(lambda x: len(x.split(' ') ) if isinstance(x, str) else 0 ) 
  df_test['nChars1'] = df_test['question1'].map(lambda x: len(x) if isinstance(x, str) else 0 ) 
  df_test['nChars2'] = df_test['question2'].map(lambda x: len(x) if isinstance(x, str) else 0 ) 
  
  #Make lists of the single double triple words
  df_test['question1'] = df_test['question1'].fillna("")
  df_test['question2'] = df_test['question2'].fillna("")
  df_test['q2_stem_noStop'] = np.empty((len(df_test), 0)).tolist()
  
    
  #For already completed columsn with blank strings being read as columns
  df_test.loc[df_test['question1'].isnull(), ['question1']] = df_test.loc[df_test['question1'].isnull(), 'question1'].apply(lambda x: [])
  df_test.loc[df_test['question2'].isnull(), ['question2']] = df_test.loc[df_test['question2'].isnull(), 'question2'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem'].isnull(), ['q1_stem']] = df_test.loc[df_test['q1_stem'].isnull(), 'q1_stem'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem'].isnull(), ['q2_stem']] = df_test.loc[df_test['q2_stem'].isnull(), 'q2_stem'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem_noStop'].isnull(), ['q1_stem_noStop']] = df_test.loc[df_test['q1_stem_noStop'].isnull(), 'q1_stem_noStop'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem_noStop'].isnull(), ['q2_stem_noStop']] = df_test.loc[df_test['q2_stem_noStop'].isnull(), 'q2_stem_noStop'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem_double'].isnull(), ['q1_stem_double']] = df_test.loc[df_test['q1_stem_double'].isnull(), 'q1_stem_double'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem_double'].isnull(), ['q2_stem_double']] = df_test.loc[df_test['q2_stem_double'].isnull(), 'q2_stem_double'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem_double_noStop'].isnull(), ['q1_stem_double_noStop']] = df_test.loc[df_test['q1_stem_double_noStop'].isnull(), 'q1_stem_double_noStop'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem_double_noStop'].isnull(), ['q2_stem_double_noStop']] = df_test.loc[df_test['q2_stem_double_noStop'].isnull(), 'q2_stem_double_noStop'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem_triple'].isnull(), ['q1_stem_triple']] = df_test.loc[df_test['q1_stem_triple'].isnull(), 'q1_stem_triple'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem_triple'].isnull(), ['q2_stem_triple']] = df_test.loc[df_test['q2_stem_triple'].isnull(), 'q2_stem_triple'].apply(lambda x: [])
  df_test.loc[df_test['q1_stem_triple_noStop'].isnull(), ['q1_stem_triple_noStop']] = df_test.loc[df_test['q1_stem_triple_noStop'].isnull(), 'q1_stem_triple_noStop'].apply(lambda x: [])
  df_test.loc[df_test['q2_stem_triple_noStop'].isnull(), ['q2_stem_triple_noStop']] = df_test.loc[df_test['q2_stem_triple_noStop'].isnull(), 'q2_stem_triple_noStop'].apply(lambda x: [])
  print ("Filled NaN")

  df_test['q1_stem'] = df_test.apply(lambda x: wordToStemmedList(x['question1']), axis=1 )
  print ("q1_stem")
  df_test['q2_stem'] = df_test.apply(lambda x: wordToStemmedList(x['question2']), axis=1 )
  print ("q2_stem")
  df_test['q1_stem_noStop'] = df_test.apply(lambda x: wordToStemmedListNoStop(x['question1']), axis=1 )
  print ("q1_stem_noStop")
  df_test['q2_stem_noStop'] = df_test.apply(lambda x: wordToStemmedListNoStop(x['question2']), axis=1 ) 
  print ("q2_stem_noStop")
 
  df_test['q1_stem_double'] = df_test.apply(lambda x: createDoubleWords(x['q1_stem']), axis=1 )
  df_test['q2_stem_double'] = df_test.apply(lambda x: createDoubleWords(x['q2_stem']), axis=1 )
  df_test['q1_stem_double_noStop'] = df_test.apply(lambda x: createDoubleWords(x['q1_stem_noStop']), axis=1 )
  df_test['q2_stem_double_noStop'] = df_test.apply(lambda x: createDoubleWords(x['q2_stem_noStop']), axis=1 )
  print ("doubles")
 
  df_test['q1_stem_triple'] = df_test.apply(lambda x: createTripleWords(x['q1_stem']), axis=1 )
  df_test['q2_stem_triple'] = df_test.apply(lambda x: createTripleWords(x['q2_stem']), axis=1 )
  df_test['q1_stem_triple_noStop'] = df_test.apply(lambda x: createTripleWords(x['q1_stem_noStop']), axis=1 )
  df_test['q2_stem_triple_noStop'] = df_test.apply(lambda x: createTripleWords(x['q2_stem_noStop']), axis=1 )
  print ("triples")

  """   
  #Calculate numbers based upon manipulated strings
  df_test['diffWords'] = df_test['nWords1'] - df_test['nWords2']
  df_test['diffChars'] = df_test['nChars1'] - df_test['nChars2']
  
  df_test['match_stem'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem'], x['q2_stem']), axis=1 )
  df_test['match_stem_noStop'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem_noStop'], x['q2_stem_noStop']), axis=1 )
  df_test['match_stem_double'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem_double'], x['q2_stem_double']), axis=1 )
  df_test['match_stem_double_noStop'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem_double_noStop'], x['q2_stem_double_noStop']), axis=1 )
  df_test['match_stem_triple'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem_triple'], x['q2_stem_triple']), axis=1 )
  df_test['match_stem_triple_noStop'] = df_test.apply(lambda x: twoStringListMatches(x['q1_stem_triple_noStop'], x['q2_stem_triple_noStop']), axis=1 )
  print ("Finished match_")
  
  df_test['nChars_match_stem'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem'], x['q2_stem']), axis=1 )
  df_test['nChars_match_stem_noStop'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_noStop'], x['q2_stem_noStop']), axis=1 )
  df_test['nChars_match_stem_double'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_double'], x['q2_stem_double']), axis=1 )
  df_test['nChars_match_stem_double_noStop'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_double_noStop'], x['q2_stem_double_noStop']), axis=1 )
  df_test['nChars_match_stem_triple'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_triple'], x['q2_stem_triple']), axis=1 )
  df_test['nChars_match_stem_triple_noStop'] = df_test.apply(lambda x: numCharsOfStringListMatches(x['q1_stem_triple_noStop'], x['q2_stem_triple_noStop']), axis=1 )
  print ("Finished nChars_match_")
  
  df_test['sequential_match_stem'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem'], x['q2_stem']), axis=1 )
  df_test['sequential_match_stem_noStop'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_noStop'], x['q2_stem_noStop']), axis=1 )
  df_test['sequential_match_stem_double'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_double'], x['q2_stem_double']), axis=1 )
  df_test['sequential_match_stem_double_noStop'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_double_noStop'], x['q2_stem_double_noStop']), axis=1 )
  df_test['sequential_match_stem_triple'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_triple'], x['q2_stem_triple']), axis=1 )
  df_test['sequential_match_stem_triple_noStop'] = df_test.apply(lambda x: longestSeriesOfWordMatches(x['q1_stem_triple_noStop'], x['q2_stem_triple_noStop']), axis=1 )
  print ("Finisehd sequential_match_")

except KeyboardInterrupt:
  print ("Ended")

#Writing the dataframe out
df_test.to_csv("testing_data/test_match.csv", index=False)
#pickle.dump(df_test, open( "testing_data/test_doublesAndTriples.p", "wb" ) )
