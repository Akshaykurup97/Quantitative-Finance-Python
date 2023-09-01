import numpy as np

def high_breakdown_fp(epsi, p, last=0, c=0.5):
    
    i_, t_ = epsi.shape
    p_tilde = np.zeros((1, 1))
    mu_HBFP = np.zeros((i_, 1))
    sigma2_HBFP = np.zeros((i_, i_, 1))
    v = np.zeros(1)
    t_out = np.zeros(1)

    # Step 0: initialize
    k = 0
    T = np.arange(t_)
    while k == 0 or p_tilde[k] >= c:

        # Step 1: probability
        p_tilde = np.r_[p_tilde, np.sum(p,keepdims=True)]
        
        # Step 2: fit high-breakdown estimators of mean and covariance 
        j = 0
        det1 = 0
        k = k + 1
        t_ = epsi.shape[1]
        w = np.ones((1, t_))/t_
        mu = np.mean(epsi, axis=1, keepdims=True)
        mu_MVE = mu
        sigma2 = np.cov(epsi)
        sigma2_MVE = sigma2[..., np.newaxis]
        keep_loop = 1  # loop condition 
        while keep_loop:
            j = j + 1
            # z-scores
            z_score = np.sum((epsi - mu)*np.linalg.solve(sigma2,(epsi - mu)), axis=0)
            # update weights
            update = np.where(z_score > 1)
            w[0,update] = w[0, update]*z_score[update]
            # update mean vector
            mu = (epsi@w.T)/np.sum(w)  
            mu_MVE = np.r_['-1', mu_MVE, mu]
            # update covariance matrix
            sigma2 = (epsi - np.tile(mu, (1, t_)))@np.diagflat(w)@(epsi - np.tile(mu, (1, t_))).T
            sigma2_MVE = np.r_['-1', sigma2_MVE, sigma2[..., np.newaxis]]
            # check convergence
            keep_loop = 1 - np.prod(z_score <= 1)     
            
        # high-breakdown estimators of mean and covariance 
        mu_HBFP = np.r_['-1',mu_HBFP, mu_MVE[:, -1][..., np.newaxis]]
        sigma2_HBFP = np.r_['-1', sigma2_HBFP, sigma2_MVE[:, :, -1][..., np.newaxis]]

        # Step 3: volume
        v = np.r_[v, np.linalg.det(sigma2_HBFP[:, :, k])]

        # Step 4: detect outliers
        # mean vector and covariance matrix
        mu = np.mean(epsi, axis=1, keepdims=True)
        sigma2 = np.cov(epsi)
        
        # normalize observations
        t_ = epsi.shape[1]
        z = np.sqrt(1/t_)*(epsi - np.tile(mu, (1, t_))).T
    
        # compute FP-info matrix
        h = z@np.linalg.solve(sigma2, z.T)
        
        # determine singularities
        h = np.diag(h).T
        # exclude singularities
        h = np.where(h == 1, 0, h)
        t_index = np.where(h == 1, 0, range(t_))
        
        # outlier probability
        p = np.where(h == 1, 0, p)
        # outlier
        a = (1 - h)/(1 - p)
        # outlier position index 
        t_tilde = t_index[a[0] == np.min(a)]

        # Step 5: remove outlier
        t_out = np.r_[t_out, T[t_tilde]]
        T = np.delete(T, t_tilde)
        epsi = np.delete(epsi, t_tilde, axis=1)
        p = np.delete(p, t_tilde, axis=1)
        print('\r{:.2f} %'.format(min(100*(1 - p_tilde[-1]).squeeze()/(1 - c), 100)), end='', flush=True)
            
    # Step 6: when p_tilde < c return the output for index k - 1 so that the enclosed probability for the output is p_tilde>=c
    # mean vectors computed at each iteration
    mu_HBFP = mu_HBFP[:, :-1]
    # covariance matrix computed at each iteration
    sigma2_HBFP = sigma2_HBFP[:, :, :-1]
    # probability
    p_tilde = p_tilde[:-1]
    # volume
    v = v[:-1]
    # outliers computed at steps k and (k-1) are included in ellipsoid so they have to be removed 
    t_out = t_out[:-2]

    if last != 0:
        return mu_HBFP[:, -1], sigma2_HBFP[:, :, -1], p_tilde[-1], v[-1], t_out[-1]
    else:
        return mu_HBFP, sigma2_HBFP, p_tilde, v, t_out
