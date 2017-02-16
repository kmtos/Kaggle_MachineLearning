############################################
########                          ##########
########   k Nearest Neighbors    ##########
########                          ##########
############################################

      #  ###    ###   #   # #######
     ##  #  #  #   #  #   #    #
    # #  ###   #   #  #   #    #
   ####  #  #  #   #  #   #    #
  #   #  #  #  #   #  #   #    #
 #    #  ###    ###    ###     #

1) Supervised Learning
2) Declares new point to be similar to nearby points
  2a) Looks at the "k" nearest points to determine new point label
3) Can be used for classification and regression 
  3a) Classification: Voting method, i.e. most votes of the k-neighbors wins
  3b) Regression: Mean of the label values of the k-neighbors
4) Has inherent bias
  4a) Locality: Nearness is similarness
  4b) inherently applies Smoothing via averaging, so suppresses any sort of local fluccuation.
  4c) All features matter equally (Can change this though)
5) Suffers from curse of dimentionality
  5a) As number of features grow, the amount of data needecd to generalize accurately groiws exponentially
6) How to judge which points are the nearest neighbors
  6a) Could use Euclidean distance with continuous variables. 
  6b) Could use discreet variables and gauge closeness by the number of mismatches
  6c) For text learning, use 'Hamming Distance'. This calculates number of substitutions needed to turn two strings of equal length to match each other.
  6d) 'Large Margin Nearest Neighbor': This creates a pseudometric by which to define 'closeness', where this local metric is distorted to make like-labeled points near and imposters far
    6d1) https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor
  6e) 'Neighbourhoos components': Addresses the problem of model selection (apparently). Complicated and don't really get it. Need to do more research, since I like what little I understood.
    6d2) https://en.wikipedia.org/wiki/Neighbourhood_components_analysis
7) What if K=n (# of neighbors == total # of points)
  7a) Makes no sense unless the distances are weighted
    7b1) Could use weighted average 1 / distance (closer points have larger impact in nearness)
    7b2) Could use locally weighted regression
      7b2a) Would need another Machine Learning algorithm to determine the regression (I find this idea VERY interesting.)


 ###  ###   ###        #  ###   ###   #    #
 #  # #  # #   #      #  #   # #   #  ##   #
 #  # #  # #   #     #   #     #   #  # #  #
 ###  ###  #   #    #    #     #   #  #  # #
 #    # #  #   #   #     #   # #   #  #   ##
 #    #  #  ###   #       ###   ###   #    #

1) Pros
  1a) Quick to learn
  1b) Very simple and easy to understand
  1c
2) Cons
  2a) Slower to query
  2b) Using wrong distance evaluator could lead to drastically incorrect classifier
    2b1) This is rare and difficult to happen. Ususally happens when certain features should be much more heavily weighted than others. 
  2c) Due to locality, nearness is assumed to be likeness (not always true), and fluccuations are suppressed.
  2d) The voting method can be flawed when the two classes are unbalanced.


 #   #  ###   #       #    #######  ###
 #   # #   #  #       #       #    #   #
 ##### #   #  #   #   #       #    #   #
 #   # #   #   # # # #        #    #   #
 #   #  ###     #   #         #     ###

1) sklearn
  1a) Parameters
    1a1) n_neighbors=5: number of neighbors
    1a2) algorithm='auto': Attempt to decide the most appropriate algorithm based. Other options include 'ball_tree' and 'kd_tree'
    1a3) metric='minkowski': Is determined with respect to 'p' parameter for distance based. Look at DistanceMetric class below
      1a3a) euclidean: sqrt( sum( (x-y)^2) )
      1a3b) chebyshev: max( |x-y| )
      1a3c) minkowski: sum( |x-y|^p) (1/p) | Must provide varaible p
      1a3d) wminkowski: sum( w*|x-y|^p) (1/p) | Must provide varaible p and the weights (w)
      1a3e) seuclidean: sqrt( (sum(x-y)^2) / V )
      1a3f) mahalanobis: Uses symobls that I'm not sure of. Research more.
      1a3g) There are more depending on various specfic situations of features (2 dimensional, all boolean, all integer). Refer to the source below.
      1a3h) http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
    1a4) p=2: 1 is Manhattan distance| 2 is Euclidean| other power follows use above in the metric provided.
    1a5) weights='uniform': Can be uniform, weighted by 'distance', or use a callable function to determine distance.
