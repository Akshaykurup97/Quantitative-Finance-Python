
import numpy as np
import statsmodels.multivariate.factor as fa


def smart_solve(sigma2, y=None, max_cond=1e15, k_=None):
    
    n_ = sigma2.shape[0]

    if y is None:
        y = np.eye(sigma2.shape[0])

    if k_ is None:
        k_ = int(max(1., 0.1 * n_))

    if np.linalg.cond(sigma2) < max_cond:
        x = np.linalg.solve(sigma2, y)
    else:
        s_vol = np.sqrt(np.diag(sigma2)) # target volatility vector
        corr_x = np.diag(1/s_vol)@sigma2@np.diag(1/s_vol)  # target correlation
        paf = fa.Factor(n_factor=k_, corr=corr_x, method='pa', smc=True).fit() # paf fitted model
        beta = paf.loadings  # paf loadings
        delta = paf.uniqueness  # paf variances
        beta = np.diag(np.sqrt(np.diag(sigma2)))@beta  # re-scaled loadings
        delta = np.diag(sigma2)*delta  # re-scaled variances

        # binomial inverse theorem
        rho2 = beta @ np.linalg.solve((beta.T / delta) @ beta + np.eye(k_), beta.T)
        x = (y.T / delta - (y.T / delta) @ rho2 / delta).T
    
    return x
