# -*- coding: utf-8 -*-

import numpy as np
from arpym.statistics.cdf_sp import cdf_sp


def cop_marg_sep(x, p=None):
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    j_, n_ = x.shape
    if p is None:
        p = np.ones(j_) / j_  # equal probabilities as default value

    # Step 1: Sort scenarios

    x_grid, ind_sort = np.sort(x, axis=0), np.argsort(x, axis=0)  # sorted scenarios

    # Step 2: Marginal cdf's

    cdf_x = np.zeros((j_, n_))
    for n in range(n_):
        cdf_x[:, n] = cdf_sp(x_grid[:, n], x[:, n], p)

    # Step 3: Copula scenarios

    u = np.zeros((j_, n_))
    for n in range(n_):
        u[ind_sort[:, n], n] = cdf_x[:, n]

    u[u >= 1] = 1 - np.spacing(1)
    u[u <= 0] = np.spacing(1)  # clear spurious outputs

    return np.squeeze(u), np.squeeze(x_grid), np.squeeze(cdf_x)
