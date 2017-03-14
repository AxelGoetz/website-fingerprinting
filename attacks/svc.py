"""
SVC attack described in "Website Fingerprinting in Onion Routing Based Anonymization Networks" and "Website Fingerprinting at Internet Scale".
However in both papers, different features were used.

A. Panchenko et al. used a radial basis function (RBF) as a kernel with C = 2^{17} and γ = 2^{-19}
For "Website Fingerprinting at Internet Scale" they used a range of values but for the purpose of this report, we will just use these values.
"""

from sklearn.svm import SVC

def get_svc(is_multiclass=True):
    """
    Depending on the thread model, we either have a multiclass problem or a non-multiclass
    """
    decision_function_shape = 'ovr' if is_multiclass else None
    return SVC(C=2**(17), kernel='rbf',gamma=2**(-19), decision_function_shape=decision_function_shape)
