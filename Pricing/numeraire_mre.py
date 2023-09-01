
import numpy as np

from arpym.views.min_rel_entropy_sp import min_rel_entropy_sp


def numeraire_mre(v_payoff, v_tnow, *, p=None, k=0):
    
    if p is None:
        j_ = v_payoff.shape[0]
        p = np.full(j_, 1/j_)

    # Step 1: Check if the selected index actually defines a numeraire

    if any(v_payoff[:, k] < 0):
        print('v_payoff[:, k] is not a numeraire.')
        return None

    # Step 2: Compute the views

    a_eq = (np.diagflat(v_payoff[:, k]**(-1))@v_payoff).T
    b_eq = v_tnow/v_tnow[k]

    # delete the restriction on the flexible probabilities that they sum to 1;
    # already embedded in min_rel_entropy_sp function

    a_eq = np.delete(a_eq, k, axis=0)
    b_eq = np.delete(b_eq, k, axis=0)

    # Step 3: Compute minimum entropy numeraire probability

    p_mre = min_rel_entropy_sp(p, None, None, a_eq, b_eq)

    # Step 4: Compute minimum entropy stochastic discount factor

    s_mre = v_tnow[k]*p_mre@(np.diagflat(v_payoff[:, k]**(-1)))@ \
        np.diagflat(p**(-1))

    return p_mre, s_mre
