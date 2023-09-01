
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def pdf_sp(h2, x_, x, p=None):
    
    if np.ndim(x_) < 2:
        x_ = x_.reshape(-1, 1).copy()
    if np.ndim(x) < 2:
        x = x.reshape(-1, 1).copy()
    j_, n_ = x.shape

    # Step 0: if the probabilities are not specified, set them to uniform

    if p is None:
        p = np.ones(j_)/j_

    # Step 1: approximate Dirac deltas via Gaussian kernel

    d = rbf_kernel(x, x_, 1./(2. * h2)) / ((2. * np.pi * h2) ** (n_ / 2.))

    # Step 2: compute scenario-probability pdf

    f = p @ d

    return f
