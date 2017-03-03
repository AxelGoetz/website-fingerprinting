"""
The following model is based on the paper "A Novel Website Fingerprinting Attack against Multi-tab Browsing Behavior" by X. Gu et al.
However unfortunately, they do not clearly outline which naive bayes implementation they used.

Since the data cannot be moddeled by a gaussian distribution nor a multivariate Bernoulli distribution, we can only assume that they used a multinomial model.
"""

from sklearn.naive_bayes import MultinomialNB

def get_naive_bayes(is_multiclass=True):
    return MultinomialNB()
