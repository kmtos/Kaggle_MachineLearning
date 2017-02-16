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


1) Supervised learning technique.
  1a) Generally a classification, not regression. Works well with 2 classes, but can do more
2) Works on linearly seperable data (Can draw a line to divide data into two (or more) classes
  2a) Can use something called a kernel trick to transform non-linearly sepearable to linearly separable

^
|       +    +   +
|  _      +  +  ++
|   -       +   +  +
|  -  _  _   ++   +
|  _  _  _
|
+------------- >

3) The line that divides the two groups of data is decided by maximizing the margin, or the distance between the two points
  3a) Wants to maximize the margin while also classifying correctly
  3b) You can adjust parameters to decide how important maximizing vs correctness is
4) Kernels
  4a) Linear K(X, Y) = X.T *  Y
  4b) RBF K(X, Y) = e^( |X - Y|^2 / 2 * sigma^2) - Guassian (Very slow in comparison to others. Especially on features that are NOT discreet
  4c) Polynomial K(X, Y) = (X.T * Y + C)^d -Can adjust with a parameter that only applies to poly kernels
  4d) Sigmoid K(X, Y) = tanh( a * X.T * Y + c) - Also called the hyperbolic tangent
  4e) Circle K(X, Y) = ( X.T *  Y )^2
  4f) Custom kernel - Only requirement is that the matrix represents some sort of distance


 ###  ###   ###        #  ###   ###   #    #
 #  # #  # #   #      #  #   # #   #  ##   #
 #  # #  # #   #     #   #     #   #  # #  #
 ###  ###  #   #    #    #     #   #  #  # #
 #    # #  #   #   #     #   # #   #  #   ##
 #    #  #  ###   #       ###   ###   #    #

1) Pros
  1a) Effective in high dimensional space, even where number of dimensions is greater than the number of samples
  1b) Memory is efficient
  1c) Extremely versatile in with the kernel functions
2) Cons
  2a) Incredibly slow (n^3)
  2b) Poor performance if number of features is much greater than the number of samples
  2c) Doesn't naturally provide probability estimates, although it can in skLearn
  2d) Also doesn't do well wiht noise and overlapping class.

 #   #  ###   #       #    #######  ###
 #   # #   #  #       #       #    #   #
 ##### #   #  #   #   #       #    #   #
 #   # #   #   # # # #        #    #   #
 #   #  ###     #   #         #     ###      

1) sklearn
  1a) Parameters
    1a1) C = 1 : Smoothness vs correctness tradeoff. Low C is more smooth.
    1a2) degree = 3: Only for poly kernel. Determines the degree eof hte polynomial
    1a3) gamma = auto: How much influence   a single training example has. Large gamma means other points have to be closer
    1a4) class_weight = None: For if one class is more abundant. Can use 'balanced' or attribute fraction to each class( {0:.1, 1:.9} ). NOTE: You want the fractions to add up to 1.
  1b) To search the parameters space use "from sklearn.model_selection import GridSearchCV" to optimize via "score"
    1b1) parameters= dict of parameters to search
    1b2) cv=None: 1)None: default 3-fold cross-validation, 2) int= # of folds.




