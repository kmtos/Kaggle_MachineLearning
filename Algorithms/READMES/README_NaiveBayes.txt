############################################
########                          ##########
########       Naive Bayes        ##########
########                          ##########
############################################

      #  ###    ###   #   # #######
     ##  #  #  #   #  #   #    #
    # #  ###   #   #  #   #    #
   ####  #  #  #   #  #   #    #
  #   #  #  #  #   #  #   #    #
 #    #  ###    ###    ###     #

1) Based upon 'Bayes Rule'
2) Good for text learning
3) Assumes value of any featur is independent of other features given the class variable
4) Uses maximum likelihood
5) Is ONLY a Supervised Learning technique
6) Good for small samples of training data
7) Must assume a distribution or generate non-parametric models (Not sure how to do or if I want to)
  7a) Gaussian: for continuous data (Most popular)
  7b) Bernoulli: features are independent boolean variables (Also known as multi-variat). Basically a decision "word does/doesn't appear in data"
  7c) Multinomial: Basically a count of Bernoulli. "Word appears in data __ times."
    7c1) If using this, incorporate a small sample correction, or if feature and class never occur together, then the probability=0 identically.


 ###  ###   ###        #  ###   ###   #    #
 #  # #  # #   #      #  #   # #   #  ##   #
 #  # #  # #   #     #   #     #   #  # #  #
 ###  ###  #   #    #    #     #   #  #  # #
 #    # #  #   #   #     #   # #   #  #   ##
 #    #  #  ###   #       ###   ###   #    #

1) Pros
  1a) Simple and easy
  1b) Good with big feature spaces
  1c) Very efficient
  1d) Get to decide a prior distribution, so it's customizable
2) Cons
  2a) Does not handle multiple words having meaning together that's different than separately (ex: Chicago Bulls)
  2b) Not the most accurate technique
  2c) Must decide a prior distribtution, and the data might not be perfectly modeled by it.

 #   #  ###   #       #    #######  ###
 #   # #   #  #       #       #    #   #
 ##### #   #  #   #   #       #    #   #
 #   # #   #   # # # #        #    #   #
 #   #  ###     #   #         #     ###

1) Sklearn
  1a) Parameters
    1a1) Naive Bayes Gaussian has no adjustable parameters.
    1a2) Multinomial
      1a2a) alpha=1.0: smoothing parameter; 0 for no smoothing.
      1a2b) fit_prior=True: Whether to learn class prior probabilities or not
    1a3) Bernoulli
       1a3a) alpha=1.0: For smoothing; 0 for no smoothing.
       1a3b) binarize=0: Threshold for binarizing (mapping to Booleans). If 'None', data is assumed to be already boolean
