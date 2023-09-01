
import numpy as np


def marchenko_pastur(lam, q, r_):
   
    # Step 0: Set the minimum allowed variance
    
    if lam < 1e-8:   # minimum allowed variance   
        lam = 1e-8   
            
    # Step 1: Find the interval delimiters  
            
    lambda_minus = lam * (1 - np.sqrt(1. / q))**2   # interval delimiters  
    lambda_plus = lam * (1 + np.sqrt(1. / q))**2     
    
    # Step 2: Compute an equally spaced grid where the density is evaluated
    
    if q > 1:  
        x_r = np.linspace(lambda_minus, lambda_plus, r_)   # equally spaced grid
        f_mp = q / (2 * np.pi * lam * x_r) * np.sqrt((x_r - lambda_minus) * (lambda_plus - x_r))   # density computation
    elif q < 1:
        x_tmp = np.linspace(lambda_minus, lambda_plus, r_-1)
        y_tmp = q / (2 * np.pi * lam * x_tmp) * np.sqrt((x_tmp - lambda_minus) * (lambda_plus - x_tmp))
        x_r = [0, x_tmp]
        f_mp = [(1-q), y_tmp]
    else:
        x_r = np.linspace(lambda_minus + 1e-9, lambda_plus, r_)
        f_mp = q / (2 * np.pi * lam * x_r) * np.sqrt((x_r - lambda_minus) * (lambda_plus - x_r))
        
    return x_r,f_mp
