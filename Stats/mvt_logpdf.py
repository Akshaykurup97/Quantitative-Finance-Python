

import numpy as np
from scipy import special
from scipy import stats


def mvt_logpdf(x, mu, sigma2, nu):
   
    if np.shape(sigma2) == 0:
        # univaraite student t
        lf = stats.t.logpdf(x, nu, mu, sigma2)
    else:
        # multivariate student t
        n_ = sigma2.shape[0]
        d2 = np.sum((x - mu).T * np.linalg.solve(sigma2, (x - mu).T), axis=0)
        lf = -((nu + n_) / 2.) * np.log(1. + d2 / nu) + \
            special.gammaln((nu + n_) / 2.) - \
            special.gammaln(nu / 2.) - \
            (n_ / 2.) * np.log(nu * np.pi) - \
            0.5 * np.linalg.slogdet(sigma2)[1]

    return lf
