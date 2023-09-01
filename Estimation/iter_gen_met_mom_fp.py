import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize


def iter_gen_met_mom_fp(eps, p, Model, Order=2):
    
    if Model=='Poisson':

        # general settings
        NmaxIter = 100
        lam = np.mean(eps)*np.ones(NmaxIter)
        conv=0
        i=1
        t_ = p.shape[1]
        # 0. Set initial weighting matrix omega_2
        omega_2 = np.eye(2)

        #Set initial vector v_lamda and initial quadratic form in omega_2
        a = (eps - lam[0])
        b = (eps**2) - lam[0]*(lam[0] + 1)
        v_lamda = np.r_[p@a.T, p@b.T]
        quadform = v_lamda.T@omega_2@v_lamda@np.ones((1, NmaxIter))

        while i<NmaxIter and conv==0:

            # 1. Update output lambda
            lam[i] = gmm_poisson(eps, p, omega_2, lam[i - 1])  #compute the new lambda

            #2. Update weighting matrix omega_2
            a = (eps - lam[i])
            b = (eps**2) - lam[i]*(lam[i] + 1)
            v_lamda = np.r_[p@a.T, p@b.T]
            rhs = np.r_[np.r_['-1', p@(a**2).T, p@(a*b).T], np.r_['-1', p@(a*b).T, p@(b**2).T]]
            omega_2 = solve(rhs, np.eye(rhs.shape[0]))

            #shrinkage towards identity of weighting matrix omega_2
            aa = np.sqrt(p@(a**2).T)
            bb = np.sqrt(p@(b**2).T)
            c = (omega_2/np.r_[np.r_['-1',aa**2, aa*bb], np.r_['-1',aa*bb, bb**2]])
            omega_2 = 0.5*np.diagflat(np.r_['-1',aa, bb])@c@np.diagflat(np.r_['-1',aa, bb]) + 0.5*np.eye(2) # new weighting matrix

            # 3. If convergence, return the output, else: go to 1
            quadform[0, i] = v_lamda.T@omega_2@v_lamda
            reldistance = abs((quadform[0, i] - quadform[0, i - 1])/quadform[0, i - 1])
            if reldistance < 10**-8:
                conv=1
            i = i + 1
        lam1 = lam[i - 1]
        
    return lam1


def gmm_poisson(eps, p, omega2, lambda0):


    options = {'maxiter' : 5000}

    # Solve the minimization problem
    lam = minimize(gmm_p, lambda0, args=(omega2, eps, p), options=options, tol=1e-8)
    return lam.x


def gmm_p(lambda0, omega2, eps, p):
    a = (eps - lambda0)
    b = (eps**2) - lambda0*(lambda0 + 1)

    v = np.r_[p@a.T, p@b.T]
    f = v.T@omega2@v
    return f.squeeze()
