
import numpy as np


def gram_schmidt(v, sigma2=None, method='fwd'):
    
    
    n_ = v.shape[0]

    # Step 0. Initialization
    if sigma2 is None:
        sigma2 = np.eye(n_)
    w = np.zeros_like(sigma2)

    p_fwd = np.zeros((n_, n_-1))
    p_bck = np.zeros((1, n_))
    
    for n in range(n_):
        v_n = v[:, [n]]

        # Step 1a. Forward projection
        if method=='fwd':
            for m in range(n):
                p_fwd[:, [m]] = (w[:, [m]].T @ sigma2 @ v_n) * w[:, [m]]
            v_n_n1 = p_fwd[:, :n].sum(axis=1).reshape(-1, 1)
            
        # Step 1b. Backward projection    
        elif method=='bck':
            g = v[:, n+1:].T @ sigma2 @ v[:, n+1:] # gramian matrix

            for m in range(n+1,n_):
                p_bck[:, m] = v[:, [m]].T @ sigma2 @ v_n
            v_n_n1 = (p_bck[:, n+1:] @ np.linalg.inv(g) @ v[:, n+1:].T).reshape(-1,1)
                
        else:
            print('Please select method fwd or bck')
            break

        # Step 2. Orthogonalization
        u_n = v_n - v_n_n1
 
        # Step 3. Normalization
        w[:, [n]] = u_n/np.sqrt(u_n.T @ sigma2 @ u_n)

    return w    
