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

