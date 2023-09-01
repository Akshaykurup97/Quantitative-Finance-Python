
import numpy as np


def naive_selection(optim, n_, k_=None):
    
    if k_:
        k_ = min(k_, n_)
    else:
        k_ = n_

    # Step 1: Optimize over 1-element selections
    f_x1 = np.zeros(n_)
    for n in range(1, n_+1):
        all = optim(np.array([n]))
        f_x1[n-1] = all[1]

    # Step 2: Naive sorting
    n_sort = np.argsort(f_x1)

    # Step 3: Initialize selection
    s_k_naive = []
    s_k_naive_list = []

    x_naive = []
    f_x_naive = np.zeros(k_)
    for k in range(k_):
        # Step 4: Build naive selection (set)
        s_k_naive = n_sort[:k+1] + 1
        s_k_naive_list.append(s_k_naive)

        # Step 5: Optimize over k-element selections
        all = optim(s_k_naive)
        x_naive.append(all[0]),
        f_x_naive[k] = all[1]

    return x_naive, f_x_naive, s_k_naive_list
