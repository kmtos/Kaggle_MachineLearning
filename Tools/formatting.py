import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys
import pandas as pd

def targetFeatureSplit( data ):
    """ 
        given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list (this should be the  quantity you want to predict)
        return targets and features as separate lists. (sklearn can generally handle both lists and numpy arrays as input formats when training/predicting)
    """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )
    return target, features

