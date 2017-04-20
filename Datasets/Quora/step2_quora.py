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

df_test = pd.read_csv('testing_data/test5_step1.csv', header=0)

df_test['q1_stem'] = df_test['q1_stem'].fillna("")
df_test['q2_stem'] = df_test['q2_stem'].fillna("")
df_test['q1_stem_noStop'] = df_test['q1_stem_noStop'].fillna("")
df_test['q2_stem_noStop'] = df_test['q2_stem_noStop'].fillna("")

import re
df_test['q1_stem'] = df_test['q1_stem'].map(lambda x: re.sub('['+string.punctuation+']', '', x) )
df_test['q2_stem'] = df_test['q2_stem'].map(lambda x: re.sub('['+string.punctuation+']', '', x) )
df_test['q1_stem_noStop'] = df_test['q1_stem_noStop'].map(lambda x: re.sub('['+string.punctuation+']', '', x) )
df_test['q2_stem_noStop'] = df_test['q2_stem_noStop'].map(lambda x: re.sub('['+string.punctuation+']', '', x) )

try:
  for row in df_test.itertuples():
    if row.Index < 147950: continue
    q1 = row.q1_stem #df_test.iloc[row.Index]['q1_stem']
    q2 = row.q2_stem #df_test.iloc[row.Index]['q2_stem']
    q1_noStop = row.q1_stem_noStop #df_test.iloc[row.Index]['q1_stem_noStop']
    q2_noStop = row.q2_stem_noStop #df_test.iloc[row.Index]['q2_stem_noStop']


################
# Double  Word
################
    word1_string_noStop = ""
    word1_string = ""
    if pd.isnull(q1):  continue
    else:      
      prevWord1 = ""
      prevWord1_noStop = ""
      for word in q1.split():
        if prevWord1 == "":
          prevWord1 = word
        else:
          word1_string = word1_string + " " + prevWord1 + word
          prevWord1 = word

      for  word in q1_noStop.split():
        if prevWord1_noStop == "":
          prevWord1_noStop = word
        else:
          word1_string_noStop = word1_string_noStop + " " + prevWord1_noStop + word
          prevWord1_noStop = word
      words1 = word1_string.lstrip()
      words1_noStop = word1_string_noStop.lstrip()
      df_test.loc[row.Index, 'q1_stem_double'] = words1
      df_test.loc[row.Index, 'q1_stem_double_noStop'] = words1_noStop

    
    word2_string_noStop = ""
    word2_string = ""
    if pd.isnull(q2):  continue
    else:
      prevWord2 = ""
      prevWord2_noStop = ""
      for word in q2.split():
        if prevWord2 == "":
          prevWord2 = word
        else:
          word2_string = word2_string + " " + prevWord2 + word
          prevWord2 = word

      for word in q2_noStop.split():
        if prevWord2_noStop == "":
          prevWord2_noStop = word
        else:
          word2_string_noStop = word2_string_noStop + " " + prevWord2_noStop + word
          prevWord2_noStop = word
      words2 = word2_string.lstrip()
      words2_noStop = word2_string_noStop.lstrip()
      df_test.loc[row.Index, 'q2_stem_double'] = words2 
      df_test.loc[row.Index, 'q2_stem_double_noStop'] = words2_noStop

################
# Triple words
################
    word1_string_noStop = ""
    word1_string = ""
    if pd.isnull(q1):  continue
    else:
      prevWord1 = ""
      prevWord1_noStop = ""
      prevWord2 = ""
      prevWord2_noStop = ""
      for word in q1.split():
        if prevWord1 == "":
          prevWord1 = word
        elif prevWord2 == "":
          prevWord2 = word
        else:
          word1_string = word1_string + " " + prevWord1 + prevWord2 + word
          prevWord1 = prevWord2
          prevWord2 = word

      for word in q1_noStop.split():
        if prevWord1_noStop == "":
          prevWord1_noStop = word
        elif prevWord2_noStop == "":
          prevWord2_noStop = word
        else:
          word1_string_noStop = word1_string_noStop + " " + prevWord1_noStop + prevWord2_noStop + word
          prevWord1_noStop = prevWord2_noStop
          prevWord2_noStop = word
      words1 = word1_string.lstrip()
      words1_noStop = word1_string_noStop.lstrip()
      df_test.loc[row.Index, 'q1_stem_triple'] = words1
      df_test.loc[row.Index, 'q1_stem_triple_noStop'] = words1_noStop


    word2_string_noStop = ""
    word2_string = ""
    if pd.isnull(q2):  continue
    else:
      prevWord1 = ""
      prevWord1_noStop = ""
      prevWord2 = ""
      prevWord2_noStop = ""
      for word in q2.split():
        if prevWord1 == "":
          prevWord1 = word
        elif prevWord2 == "":
          prevWord2 = word
        else:
          word2_string = word2_string + " " + prevWord1 + prevWord2 + word
          prevWord1 = prevWord2
          prevWord2 = word

      for word in q2_noStop.split():  
        if prevWord1_noStop == "":
          prevWord1_noStop = word
        elif prevWord2_noStop == "":
          prevWord2_noStop = word
        else:
          word2_string_noStop = word2_string_noStop + " " + prevWord1_noStop + prevWord2_noStop + word
          prevWord1_noStop = prevWord2_noStop
          prevWord2_noStop = word
      words2 = word2_string.lstrip()
      words2_noStop = word2_string_noStop.lstrip()
      df_test.loc[row.Index, 'q2_stem_triple'] = words2
      df_test.loc[row.Index, 'q2_stem_triple_noStop'] = words2_noStop
 
    print(row.Index)
    print (df_test.iloc[row.Index]['q1_stem'], "\n\t", df_test.iloc[row.Index]['q1_stem_noStop'], "\n\t", df_test.iloc[row.Index]['q1_stem_double'], "\n\t", df_test.iloc[row.Index]['q1_stem_double_noStop'], "\n\t", df_test.iloc[row.Index]['q1_stem_triple'], "\n\t", df_test.iloc[row.Index]['q1_stem_triple_noStop'] )
    print (df_test.iloc[row.Index]['q2_stem'], "\n\t", df_test.iloc[row.Index]['q2_stem_noStop'], "\n\t", df_test.iloc[row.Index]['q2_stem_double'], "\n\t", df_test.iloc[row.Index]['q2_stem_double_noStop'], "\n\t", df_test.iloc[row.Index]['q2_stem_triple'], "\n\t", df_test.iloc[row.Index]['q2_stem_triple_noStop'] )

except KeyboardInterrupt:
  print ("Ending Index=", row.Index)
except ValueError:
  print("Could not convert data to a string")

df_test.to_csv("testing_data/test5_step2.csv", index=False) 
