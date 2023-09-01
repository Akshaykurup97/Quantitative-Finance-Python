
import numpy as np
from scipy.optimize import minimize
import warnings

from arpym.pricing.nelson_siegel_yield import nelson_siegel_yield


def fit_nelson_siegel_bonds(v_bond, c, upsilon, *, facev=1, theta_0=None):
    
    n_ = len(v_bond)

    def fit_nelsonsiegel_bonds_target(theta):                      
        v_bond_theta = np.zeros(n_)
        output = 0.0
        for n in range(n_):
            
            # Step 1: Compute Nelson-Siegel yield curve
            
            y_ns_theta = nelson_siegel_yield(upsilon[n], theta)
             
            # Step 2: Compute coupon bond value
            
            # zero-coupon bond value            
            v_zcb = np.exp(-upsilon[n]*y_ns_theta)
            # bond value
            v_bond_theta[n] = facev * (c[n]@v_zcb)
            
            # Step 3: Compute minimization function 
            
            if n==0:
                h_tilde = (upsilon[n+1][-1]-upsilon[n][-1])/2
            elif n==n_-1:
                h_tilde = (upsilon[n][-1]-upsilon[n-1][-1])/2
            else:
                h_tilde = (upsilon[n+1][-1]-upsilon[n-1][-1])/2 
            output += h_tilde * np.abs(v_bond_theta[n] - v_bond[n])
        return output
    
    if theta_0 is None:
        theta_0 = 0.1*np.ones(4)
        
    # Step 4: Fit Nelson-Siegel parameters

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(fit_nelsonsiegel_bonds_target, theta_0,
                       bounds=((None, None), (None, None), (None, None), (0, None)))
    theta = res.x
    # Output
    return theta
