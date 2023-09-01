

import numpy as np


def backward_selection(optim, n_, k_=None):
   

    if k_:
        k_ = min(k_, n_)
    else:
        k_ = n_
    x_bwd = []
    f_x_bwd = []

    # Step 0: Initialize
    s_k_bwd = np.arange(1, n_+1)
    s_k_bwd_list = []
    i_k_bwd = [s_k_bwd]

    for k in range(n_, 0, -1):
        x_k = []
        f_x_k = []
        for s_k_i in i_k_bwd:
            # Step 1: Optimize over constraint set
            all = optim(s_k_i)
            x_k.append(all[0])
            f_x_k.append(all[1])

        # Step 2: Perform light-touch search
        opt_indices = np.argmin(f_x_k)
        s_k_bwd = i_k_bwd[opt_indices]
        if k <= k_:
            x_bwd.insert(0, x_k[opt_indices])
            f_x_bwd.insert(0, f_x_k[opt_indices])
            s_k_bwd_list.insert(0, s_k_bwd)

        # Step 3: Build (k-1)-element set of selections
        i_k_bwd = []
        for n in s_k_bwd:
            i_k_bwd.append(np.setdiff1d(s_k_bwd, n).astype(int))

    return x_bwd, np.array(f_x_bwd), s_k_bwd_list
