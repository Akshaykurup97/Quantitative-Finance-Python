

import numpy as np


def rollvalue_value(vroll, tend, tau, dates):
   

    n_ = tend.shape[0]
    t_ = dates.shape[0]
    tau_t = np.zeros([t_, n_])
    v = np.zeros([t_, n_])

    for n in range(n_):
        # compute time to maturity at each date (in years)
        tau_t[:, n] = (tend[n] - dates).astype('timedelta64[D]') /\
            np.timedelta64(360, 'D')
        for t in range(t_):
            if tau_t[t, n] >= 0:
                # for dates before the maturity,
                # obtain the value of bond via interpolation
                index_non_nan = np.where(~np.isnan(vroll[t, :]))
                v[t, n] = np.interp(tau_t[t, n],
                                    tau[index_non_nan],
                                    vroll[t, index_non_nan].flatten())
                del index_non_nan
            else:
                # for dates after the maturity, set the value to nan
                v[t, n] = np.nan
    return v
