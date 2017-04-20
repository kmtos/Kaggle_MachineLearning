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

pd.set_option('display.max_columns', 100000000000)
pd.set_option('display.max_rows', 1000000000000)


df_test = pd.read_csv('testing_data/test7.csv', header=0)
#df_test = pd.read_csv('test.csv', header=0)

#print(df_test.shape)
#print(df_test.shape)
#for col in df_test:
#  print ("column=" , col )

df_test['nWords1'] = df_test['question1'].map(lambda x: len(x.split(' ') ) if isinstance(x, str) else -1 ) 
df_test['nWords2'] = df_test['question2'].map(lambda x: len(x.split(' ') ) if isinstance(x, str) else -1 ) 
df_test['nChars1'] = df_test['question1'].map(lambda x: len(x) if isinstance(x, str) else -1 ) 
df_test['nChars2'] = df_test['question2'].map(lambda x: len(x) if isinstance(x, str) else -1 ) 

from nltk.stem.snowball import SnowballStemmer
import nltk
STOP_WORDS = nltk.corpus.stopwords.words("english")
print (STOP_WORDS)
stemmer = SnowballStemmer("english")

df_test['q1_stem'] = np.NaN
df_test['q2_stem'] = np.NaN
df_test['q1_stem_noStop'] = np.NaN
df_test['q2_stem_noStop'] = np.NaN

df_test['q1_stem_double'] = np.NaN
df_test['q2_stem_double'] = np.NaN
df_test['q1_stem_double_noStop'] = np.NaN
df_test['q2_stem_double_noStop'] = np.NaN

df_test['q1_stem_triple'] = np.NaN
df_test['q2_stem_triple'] = np.NaN
df_test['q1_stem_triple_noStop'] = np.NaN
df_test['q2_stem_triple_noStop'] = np.NaN

try:
  for row in df_test.itertuples():
    #if row.Index <= 28607: continue
    question1 = row.question1
    question2 = row.question2
  
    word1_string_noStop = ""
    word1_string = ""
    if pd.isnull(question1):  continue
    else:      
      for word in question1.split():
        word1_string = word1_string + " " + (stemmer.stem(word))
        if word in STOP_WORDS:
          continue
        else:
          word1_string_noStop = word1_string_noStop +" "+(stemmer.stem(word))
      words1 = word1_string.lstrip()
      words1_noStop = word1_string_noStop.lstrip()
      df_test.loc[row.Index, 'q1_stem'] = words1
      df_test.loc[row.Index, 'q1_stem_noStop'] = words1_noStop
    
    word2_string_noStop = ""
    word2_string = ""
    if pd.isnull(question2): continue
    else:
      for word in question2.split():
        word2_string = word2_string + " " + (stemmer.stem(word))
        if word in STOP_WORDS:
          continue
        else:
          word2_string_noStop = word2_string_noStop +" "+(stemmer.stem(word))
      words2 = word2_string.lstrip()
      words2_noStop = word2_string_noStop.lstrip()
      df_test.loc[row.Index, 'q2_stem'] = words2
      df_test.loc[row.Index, 'q2_stem_noStop'] = words2_noStop
      print ("\nDONE row.Index", row.Index )
  
    print (df_test.iloc[row.Index]['question1'], "\t", df_test.iloc[row.Index]['q1_stem'], "\t", df_test.iloc[row.Index]['q1_stem_noStop'])
    print (df_test.iloc[row.Index]['question2'], "\t", df_test.iloc[row.Index]['q2_stem'], "\t", df_test.iloc[row.Index]['q2_stem_noStop'] )

except KeyboardInterrupt:
  print ("Ending Index=", row.Index)
except ValueError:
  print("Could not convert data to a string")


df_test.to_csv("testing_data/test7_step1.csv", index=False)

