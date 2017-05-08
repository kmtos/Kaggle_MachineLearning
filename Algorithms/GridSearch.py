##########################################
# GridSearchCV to find best parameters
##########################################
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import sys


def gridSearch(estimator, features=None, labels=None, parameters=None, cross_validation=5):
    if features is None:
        print('Give me some data')
        return None
    if parameters is None:
        print('Give me some parameters to iterate over.')
        return None
    for (key, value) in parameters.items():
        print('\t', key, '=', value)

    print('##################################')
    clf = GridSearchCV(estimator=estimator, param_grid = parameters, cv = cross_validation)
    clf.fit(features, labels)
    print('BestScore=', clf.best_score_, '\nbest Parameters=', clf.best_params_)
    return clf.best_params_

