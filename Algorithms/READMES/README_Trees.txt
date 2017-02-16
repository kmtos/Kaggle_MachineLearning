############################################
########                          ##########
########  Support Vector Machines ##########
########                          ##########
############################################

      #  ###    ###   #   # #######
     ##  #  #  #   #  #   #    #
    # #  ###   #   #  #   #    #
   ####  #  #  #   #  #   #    #
  #   #  #  #  #   #  #   #    #
 #    #  ###    ###    ###     #


1) Supervised learning for Regression or Classification
  1a) Labels need to discreet values (continuous) for Classification (Regression)
2) Basically picks best feature, splits data, and then finds the next best node. Continues process until end requirements are met.
3) Entropy (E) is one way to determines where to split
  3a) E = SUM=p_i * log_2(p_i)
  3b) pi is the fraction of examples in class i at particular node
  3c) If all the examples are the same, then class E=0.0
  3d) If examples are evenly split between the classes, then E=1.0
4) Decision Trees maximize information gain=E(parent) - (weighted average) * E(children)
  4a) The weighted average can be the weighted number of total examples



 ###  ###   ###        #  ###   ###   #    #
 #  # #  # #   #      #  #   # #   #  ##   #
 #  # #  # #   #     #   #     #   #  # #  #
 ###  ###  #   #    #    #     #   #  #  # #
 #    # #  #   #   #     #   # #   #  #   ##
 #    #  #  ###   #       ###   ###   #    #

Pros
 1) Simple and easily visualized
 2) Requires little data preparation, in that other techniques often require data normalization, and blank values removed, and other stuff.
 3) The cost of using the tree is logarithmic in the number of data points used to train the tree
 4) Uses both numerical and categorical data
 5) Able to handle multi-output problems
 6) Easily statistically verified

Cons
 1) Easy to overfit (Consider boosting)
 2) Small variations in the data might result in a different tree being generated. (Mitigated by using decision trees within an ensemble.)
 3) Locally optimal decision cannot guarantee globally optimal decision trees. (Mitigated by training multiple trees in an ensemble learner, where the features and 
    samples are randomly sampled with replacement)
 4) Trees can be biased if some classes dominate. Balance the dataset prior to fitting the decision tree.


 ####### #####  ###   ###
    #      #    #  # #
    #      #    #  #  ##
    #      #    ###     #
    #      #    #       #
    #    #####  #    ###

1) Large number of features and small number of samples is very likely to overfit.
  1a) Consider using dimensionality reduction (PCA, ICA, or feature selection) beforehand.
2) Use max_depth to prevent overfitting
3) Balance the classes beforehand to prevent biasing. 
  3a) Done via sampling an equal number of both classes, or preferably by normalizing the sum of the sample weights for each class to the same value.


 #   #  ###   #       #    #######  ###
 #   # #   #  #       #       #    #   #
 ##### #   #  #   #   #       #    #   #
 #   # #   #   # # # #        #    #   #
 #   #  ###     #   #         #     ###

1) sklearn
  1a) Parameters
    1a1) max_features=None
    1a2) max_depth=None: Max depth of tree, or max number of splits.
    1a3) min_samples_split=2: Minimum number of samples required to split at a node
    1a4) min_samples_leaf=1: Minimum number of samples required to be a leaf node.
    1a5) class_weight=None: Weights associated with classes in the form {class_label: weight}. Balanced makes weights inversely proportional its frequency. 
    1a6) min_weight_fraction_leaf=0.: The minimum weighted fraction of the sum total of weights required to be a leaf node.
    1a7) criterion='gini': Can also use 'entropy'. Used to determine how to calculate information gain.
  1b) To search the parameters space use "from sklearn.model_selection import GridSearchCV" to optimize via "score"
    1b1) parameters= dict of parameters to search
    1b2) cv=None: 1)None: default 3-fold cross-validation, 2) int= # of folds.




