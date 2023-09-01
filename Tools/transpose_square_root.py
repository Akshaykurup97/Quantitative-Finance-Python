
import numpy as np

from arpym.tools.cpca_cov import cpca_cov
from arpym.tools.pca_cov import pca_cov
from arpym.tools.gram_schmidt import gram_schmidt


def transpose_square_root(sigma2, method='Riccati', d=None, v=None):
   
    n_ = sigma2.shape[0]

    if np.ndim(sigma2) < 2:
        return np.squeeze(np.sqrt(sigma2))

    method = method.lower()

    if method == 'cpca' and d is None:
        method = 'pca'
        
    # Step 1: Riccati
    if method == 'riccati':
        e, lam = pca_cov(sigma2)
        s = e @ np.diag(np.sqrt(lam)) @ e.T
        
    # Step 2: Conditional principal components
    elif method == 'cpca':
        e_d, lam_d = cpca_cov(sigma2, d)
        s = np.linalg.inv(e_d).T @ np.diag(np.sqrt(lam_d))

    # Step 3: Principal components
    elif method == 'pca':
        e, lam = pca_cov(sigma2)
        s = e @ np.diag(np.sqrt(lam))

    # Step 4: Gram-Schmidt
    elif method == 'gram-schmidt':
        if v is None:
            v = np.eye(n_)
        
        w = gram_schmidt(v, sigma2)
        s = np.linalg.inv(w).T

    # Step 5: Cholesky
    elif method == 'cholesky':
        s = np.linalg.cholesky(sigma2)

    return s
