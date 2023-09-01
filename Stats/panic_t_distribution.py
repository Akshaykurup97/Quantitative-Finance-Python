

import numpy as np
from scipy import stats

from arpym.statistics.simulate_t import simulate_t
from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp


def panic_t_distribution(nu, rho2, r, c, j_):
   
    k_ = rho2.shape[0]
    
    # Step 1: Calm Component
    z_calm = simulate_t(np.zeros(k_), rho2, nu, j_)

    # Step 2: Panic Component
    rho2_panic = (1-r) * np.eye(k_) + r * np.ones((k_, k_))
    z_panic = simulate_t(np.zeros(k_), rho2_panic, nu, j_)
    
    # Step 3: Panic trigger
    b = (z_panic < stats.t.ppf(c, nu)) # triggers panic
    
    # Step 4: Panic scenarios
    z = (1-b) * z_calm + b * z_panic

    # Step 5: MRE flexible probabilities
    p_pri = np.ones(j_,) / j_ # flat flexible probabilities (prior)
    aeq = np.vstack((np.ones((1, j_)), z.T))
    beq = np.vstack((np.array([1]), np.zeros((k_,1))))

    p = min_rel_entropy_sp(p_pri, z_eq=aeq , mu_view_eq=np.squeeze(beq)) # posterior probabilities
    
    # Step 6: Panic FP distribution
    return z, p


