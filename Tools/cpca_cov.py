
import numpy as np

from arpym.tools.pca_cov import pca_cov


def cpca_cov(sigma2, d, old=False):
    

    n_ = sigma2.shape[0]
    k_ = d.shape[0]
    i_n = np.eye(n_)
    lam_d = np.empty((n_, 1))
    e_d = np.empty((n_, n_))

    # Step 0. initialize constraints
    m_ = n_ - k_
    a_n = np.copy(d)

    for n in range(n_):
        # Step 1. orthogonal projection matrix
        p_n = i_n-a_n.T@np.linalg.inv(a_n@a_n.T)@a_n

        # Step 2. conditional dispersion matrix
        s2_n = p_n @ sigma2 @ p_n

        # Step 3. conditional principal directions/variances
        e_d[:, [n]], lam_d[n] = pca_cov(s2_n, 1)

        # Step 4. Update augmented constraints matrix
        if n+1 <= m_-1:
            a_n = np.concatenate((a_n.T, sigma2 @ e_d[:, [n]]), axis=1).T
        elif m_ <= n+1 <= n_-1:
            a_n = (sigma2 @ e_d[:, :n+1]).T

    return e_d, lam_d.squeeze()
