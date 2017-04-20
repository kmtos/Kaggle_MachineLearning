import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import math

##################################
# Returns a list of stemmed words
##################################
def wordToStemmedList(question):
  if question == '': return ''
  question = question.lower()
  stemmer = SnowballStemmer("english")
  stemmedList = []
  list1 = question.split(" ")
  for word in list1:
    word = re.sub('['+string.punctuation+']', '', word)
    stemmedList.append(stemmer.stem(word) )
  return stemmedList

####################################################
# Returns a list of stemmed words with no Stopwords
####################################################
def wordToStemmedListNoStop(question):
  if question == '': return ''
  question = question.lower()
  stemmer = SnowballStemmer("english")
  STOP_WORDS = nltk.corpus.stopwords.words("english")
  stemmedList = []
  list1 = question.split(" ")
  for word in list1:
    if word not in STOP_WORDS:
      word = re.sub('['+string.punctuation+']', '', word)
      stemmedList.append(stemmer.stem(word) )
  return stemmedList

#################################################################
# Creates words of pairs of neighboring words from list of words
#################################################################
def createDoubleWords(list1):
  doubleList = []
  prevWord = ''
  if list1 is np.NaN: 
    return [''] 
  list1 = re.sub('['+string.punctuation+']', '', list1)
  list1 = list1.split(" ")
  for word in list1:
    if prevWord == '': prevWord = word
    else:  
      doubleList.append(prevWord + word)
      prevWord = word
  return doubleList

###################################################################
# Creates words of triples of neighboring words from list of words
###################################################################
def createTripleWords(list1):
  tripleList = []
  prevWord1 = ''
  prevWord2 = ''
  if list1 is np.NaN: 
    return ['']
  list1 = re.sub('['+string.punctuation+']', '', list1)
  list1 = list1.split(" ")
  for word in list1:
    if prevWord1 == '': prevWord1 = word
    elif prevWord2 == '': prevWord2 = word
    else:
      tripleList.append(prevWord1 + prevWord2 + word)
      prevWord1 = prevWord2
      prevWord2 = word
  return tripleList


##########################################################
#  Coutns number of matvches between two lists of numbers
##########################################################
def twoStringListMatches(list1, list2):
  if list1 is np.NaN or list2 is np.NaN:
    return 0
  list1 = re.sub('['+string.punctuation+']', '', list1)
  list1 = list1.split(" ")
  list2 = re.sub('['+string.punctuation+']', '', list2)
  list2 = list2.split(" ")
  nMatches1 = 0
  for word in list1:
    if word in list2:
      nMatches1 += 1
  nMatches2 = 0
  for word in list2:
    if word in list1:
      nMatches2 += 1
  if len(list1) + len(list2) == 0: 
    return 0
  return (nMatches1 + nMatches2) / (1.0 * (len(list1) + len(list2)) )

#############################################################
# Counts summed length of words that match between two lists
#############################################################
def numCharsOfStringListMatches(list1, list2):
  if list1 is np.NaN or list2 is np.NaN:
    return 0
  list1 = re.sub('['+string.punctuation+']', '', list1)
  list1 = list1.split(" ")
  list2 = re.sub('['+string.punctuation+']', '', list2)
  list2 = list2.split(" ")
  charSum1 = 0
  charSumTotal = 0
  for word in list1:
    charSumTotal += len(word)
    if word in list2:
      charSum1 += len(word)
  charSum2 = 0
  for word in list2:
    charSumTotal += len(word)
    if word in list1:
      charSum2 += len(word)
  if charSumTotal == 0: 
    return 0
  return (charSum1 + charSum2) / (charSumTotal * 1.0)

###############################################################################
# Finds longest number of consecutively matched words between two string lists
###############################################################################
def longestSeriesOfWordMatches(list1, list2):
  if list1 is np.NaN or list2 is np.NaN:
    return 0
  list1 = re.sub('['+string.punctuation+']', '', list1)
  list1 = list1.split(" ")
  list2 = re.sub('['+string.punctuation+']', '', list2)
  list2 = list2.split(" ")
  currentNumWordsInARow = 0
  bestNumWordsInARow = 0
  for word in list1:
    if word in list2:
      wordIndex1 = list1.index(word)
      wordIndex2 = list2.index(word)
      currentNumWordsInARow = 1
      for ite in range(1, len(list1) - wordIndex1):
        if wordIndex2 + ite >= len(list2): break
        if list1[wordIndex1 + ite] == list2[wordIndex2 + ite]:
          currentNumWordsInARow += 1
      if currentNumWordsInARow > bestNumWordsInARow: 
        bestNumWordsInARow = currentNumWordsInARow
  if (len(list1) + len(list2) ) == 0:
    return 0
  return bestNumWordsInARow / ((len(list1) + len(list2)) / 2.0)
