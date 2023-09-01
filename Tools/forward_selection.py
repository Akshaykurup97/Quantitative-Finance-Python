

import numpy as np


def forward_selection(optim, n_, k_=None):
   

    if k_:
        k_ = min(k_, n_)
    else:
        k_ = n_
    i_1 = np.arange(1, n_+1)
    x_fwd = []
    f_x_fwd = np.zeros(k_)

    # Step 0: Initialize
    s_k_fwd = []
    s_k_fwd_list = []

    for k in range(k_):
        # Step 1: Build k-element set of selections
        s_prev_fwd = s_k_fwd
        i_k_fwd = []
        x_k = []
        f_x_k = []
        for n in np.setdiff1d(i_1, s_prev_fwd):
            i_k_fwd.append(np.union1d(s_prev_fwd, n).astype(int))

            # Step 2: Optimize over constraint set
            all = optim(i_k_fwd[-1])
            x_k.append(all[0])
            f_x_k.append(all[1])

        # Step 3: Perform light-touch search
        opt_indices = np.argmin(f_x_k)
        x_fwd.append(x_k[opt_indices])
        f_x_fwd[k] = f_x_k[opt_indices]
        s_k_fwd = i_k_fwd[opt_indices]
        s_k_fwd_list.append(s_k_fwd)

    return x_fwd, f_x_fwd, s_k_fwd_list
