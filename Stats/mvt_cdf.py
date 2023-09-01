
import numpy as np
import scipy 
from scipy.stats import multivariate_normal as mvn
from arpym.statistics.q_chi import q_chi

def mvt_cdf(y,mu,sigma2,nu,du):
   
    
    # Step 0: Compute volatility vector and correlation matrix
    
    n_ = sigma2.shape[0]  # dimension 
    k_ = y.shape[0]  # number of points
    y = np.atleast_2d(y)
    sigma_vol = np.sqrt(np.diag(sigma2)).reshape(n_,1)  # volatility vector
    cor = np.linalg.pinv(np.diagflat(sigma_vol))@sigma2@np.linalg.pinv(np.diagflat(sigma_vol))  # correlation of normal distribution
    mean = np.zeros((n_,))  # mean of normal distribution 
    dist = mvn(mean=mean, cov=cor)
    
    # Step 1: if n>70 calculate cdf
    
    if nu > 70:
        Ft = np.array([dist.cdf(y_k) for y_k in y])
    else:
        
    # Step 2: Calculate grid
    
        U = np.arange(0,1,du)
            
    # Step 3: Calculate the grid of smooth quantiles
    
        q = []
        for k in range(len(U)):
            q.append(q_chi(nu,U[k]))  # grid of smooth quantiles
                        
    # Step 4: Compute numerical approximation of cdf
   
        x = np.array([(y_k-mu)/sigma_vol.reshape(n_,) for y_k in y]).reshape(k_,n_)
        Ft = np.array([du * sum([dist.cdf(q_i*x_k) for q_i in q]) for x_k in x])
    return np.squeeze(Ft)
