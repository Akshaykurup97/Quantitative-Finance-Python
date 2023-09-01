

import numpy as np
from numpy.linalg import matrix_rank, solve, eig, svd


def low_rank_diag_conditional_corr(c2, d, k_):
   
    conditional = 1
    if np.sum(abs(d.flatten())) == 0:
        conditional = 0

    n_ = c2.shape[0]

    if k_ > n_ - matrix_rank(d):
        raise Warning('k_ has to be <= rho.shape[0]-rank[d]')

    eps1 = 1e-9
    eta = 0.01
    gamma = 0.1
    constraint = 0

    #initialize output
    c2_lrd = c2
    dist = np.zeros
    n_iter = 0

    #0. Initialize
    diag_lam2, e = eig(c2)
    lam2 = diag_lam2
    lam2_ord, order = np.sort(lam2)[::-1], np.argsort(lam2)[::-1]
    lam = np.real(np.sqrt(lam2_ord[:k_]))
    e_ord = e[:, order]

    beta = np.real(e_ord[:n_,:k_]@np.diagflat(np.maximum(lam, eps1)))
    c = c2

    for j in range(1000):
        # conditional PC
        a = c - np.eye(n_) + np.diagflat(np.diag(beta@beta.T))
        if conditional == 1:
            lam2, e = conditional_pc(a, d)
            lam2 = lam2[:k_]
            e = e[:,:k_]
            lam = np.sqrt(lam2)
        else:
            # if there aren't constraints: standard PC using the covariance matrix
            diag_lam2, e1 = eig(a)
            lam2 = diag_lam2
            lam2_ord, order = np.sort(lam2)[::-1], np.argsort(lam2)[::-1]
            e_ord = e1[:, order]
            e = e_ord[:,:k_]
            lam = np.sqrt(lam2_ord[:k_])

        # loadings
        beta_new = e@np.diagflat(np.maximum(lam, eps1))
        # rows length
        l_n = np.sqrt(np.sum(beta_new**2, 1))
        # rows scaling
        beta_new[l_n > 1,:] = beta_new[l_n > 1, :]/np.tile(l_n[l_n > 1, np.newaxis]*(1 + gamma), (1, k_))
        # reconstruction
        c = beta_new@beta_new.T + np.eye(n_, n_) - np.diag(np.diag(beta_new@beta_new.T))
        # check for convergence
        distance = 1/n_*np.sum(np.sqrt(np.sum((beta_new - beta)**2, 1)))
        if distance <= eta:
            c2_lrd = c
            dist = distance
            n_iter = j
            beta = beta_new.copy()
            if d.shape == (1, 1):
                tol = np.max(abs(d*beta))
            else:
                tol = np.max(abs(d.dot(beta)))
            if tol < 1e-9:
                constraint = 1
                break
        else:
            beta = beta_new.copy()
            beta = np.real(beta)
            c2_lrd = np.real(c2_lrd)
            c2_lrd = (c2_lrd + c2_lrd.T)/2
    return c2_lrd, beta, dist, n_iter, constraint


def conditional_pc(sigma2, d):
   
    # general settings
    n_ = sigma2.shape[0]
    m_ = n_ - matrix_rank(d)
    lam2_d = np.empty((n_, 1))
    e_d = np.empty((n_, n_))

    # 0. initialize constraints
    a_n = d

    for n in range(n_):

        # 1. orthogonal projection matrix

        p = np.eye(n_) - a_n.T.dot(solve(a_n@a_n.T, a_n))

        # 2. conditional dispersion matrix
        s2 = p@sigma2@p

        # 3. conditional principal directions/variances

        w, v = eig(s2)
        ind = np.argsort(-w)
        eigvec, eigval = v[:, ind], w[ind]
        _,_,eigvec = svd(s2)
        eigvec = eigvec.T
        p, d = eigvec.shape
        rowidx = np.array(np.argmax(abs(eigvec), axis=0))
        colidx = np.arange(0, d)
        colsign = np.sign(eigvec[rowidx, colidx])
        eigvec = eigvec*colsign[np.newaxis,...]
        e_d[:, n] = eigvec[:, 0]
        lam2_d[n] = eigval[0]

    return lam2_d, e_d
