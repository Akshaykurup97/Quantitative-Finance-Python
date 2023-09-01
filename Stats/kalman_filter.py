import numpy as np


def kalman_filter(x_t, alpha_hat, beta_hat, s2_hat, alpha_hat_h, beta_hat_h,
                  s2_hat_h, h_t=None, p2=None):
    
    if len(x_t.shape) == 1:
        x_t = x_t.reshape(-1, 1).copy()
    t_, n_ = x_t.shape
    k_ = len(alpha_hat_h)

    # Step 1: Set the initial value h_t[:,0] for the hidden factors
    if h_t is None:
        h_t = x_t[[0], :]@beta_hat
    else:
        h_t = h_t.reshape((1, k_))

    # Step 2: Set the initial covariance of h_t[:,0], expressing the uncertainty of measurement
    if p2 is None:
        p2 = 10*np.eye(k_).reshape(k_, k_, 1)
    else:
        p2 = p2.reshape(k_, k_, 1)

    for t in range(1, t_):

        # Step 3: Estimate step
        h_t_tilde = alpha_hat_h + beta_hat_h@h_t[t-1, :]
        p2_tilde = beta_hat_h@p2[:, :, t-1]@beta_hat_h.T + s2_hat_h

        # Step 4: Correction step
        u = x_t[[t], :] - alpha_hat - beta_hat@h_t_tilde
        s2_hat_u = s2_hat + beta_hat@p2_tilde@beta_hat.T
        kappa = p2_tilde@beta_hat.T.dot(np.linalg.pinv(s2_hat_u))
        h_t = np.r_[h_t, h_t_tilde + u@kappa.T]
        p2 = np.r_['-1', p2, ((np.eye(k_) - kappa@beta_hat)@p2_tilde)[...,
                   np.newaxis]]

    return np.squeeze(h_t)
