"""
sstools is a small module containing implementation of basic state space tools
in Python.
"""

import numpy as np

def kalman_filter(y, a_init, P_init, Z, H, T, R, Q):
    """
    Perform Kalman filtering on the data y. 
    Conventions are as in Durbin and Koopman (2012).
    Relevant dimensions are:
    Nt: number of time points
    Np: dimension of observation space
    Nm: dimension of state space
    Nr: dimension of state noise covariance

    Parameters:
    a_init: mean of prior on states: (Nm, 1) or (Nm,)
    P_init: variance of prior on states: (Nm, Nm)
    Z: either a (Nt, Np, Nm) array or a (Np, Nm) array
    H: either a (Nt, Np, Np) array or a (Np, Np) array
    T: either a (Nt, Nm, Nm) array or a (Nm, Nm) array
    R: either a (Nt, Nm, Nr) array or a (Nm, Nr) array
    Q: either a (Nt, Nr, Nr) array or a (Nr, Nr) array

    Returns:
    v: (Nt, Np) vector of residuals at each time
    K: (Nt, Nm, Np) Kalman gain matrix
    Finv: (Nt, Np, Np) inverse of prediction variance matrix
    """
    # get dimensions
    Nt, Np = y.shape
    Nm = P_init.shape[0]

    # preallocate arrays
    a = np.empty((Nt, Nm))
    P = np.empty((Nt, Nm, Nm))
    # aa = np.empty((Nt, Nm))
    # PP = np.empty((Nt, Nm, Nm))
    v = np.empty((Nt, Np))
    Finv = np.empty((Nt, Np, Np))
    K = np.empty((Nt, Nm, Np))
    
    # initialize
    a[0] = a_init
    P[0] = P_init
    
    # iterate
    for t in range(Nt):
        if Z.ndim == 2:
            ZZ = Z
        else:
            ZZ = Z[t]

        if H.ndim == 2:
            HH = H
        else:
            HH = H[t]

        if T.ndim == 2:
            TT = T
        else:
            TT = T[t]

        if R.ndim == 2:
            RR = R
        else:
            RR = R[t]

        if Q.ndim == 2:
            QQ = Q
        else:
            QQ = Q[t]

        v[t] = y[t] - ZZ.dot(a[t])
        F = Z.dot(P[t]).dot(Z.T) + HH
        Finv[t] = np.linalg.inv(F)
        # aa[t] = a[t] + P[t].dot(Z.T).dot(Finv[t]).dot(v[t])
        # PP[t] = P[t] - P[t].dot(Z.T).dot(Finv[t]).dot(Z).dot(P[t])
        K[t] = TT.dot(P[t]).dot(Z.T).dot(Finv[t])
        L = TT - K[t].dot(Z)
        if t + 1 < Nt:
            a[t + 1] = TT.dot(a[t]) + K[t].dot(v[t])
            P[t + 1] = TT.dot(P[t]).dot(L.T) + RR.dot(QQ).dot(RR.T)
        
    return v, K, Finv

def fast_state_smoother(v, K, Finv, a_init, P_init, Z, T, R, Q):
    """
    Runs an efficient smoothing algorithm to get posterior mean of state 
    vector.

    Parameters are as in kalman_filter.
    """
    # infer dimensions
    Ny, Np = v.shape
    Nm = K.shape[1]
    
    # preallocate
    r = np.empty((Ny, Nm))
    r[-1] = 0
    alpha = np.empty((Ny, Nm))
    
    # calculate r
    for t in range(Ny - 1, -1, -1):
        if Z.ndim == 2:
            ZZ = Z
        else:
            ZZ = Z[t]

        if T.ndim == 2:
            TT = T
        else:
            TT = T[t]

        if R.ndim == 2:
            RR = R
        else:
            RR = R[t]

        if Q.ndim == 2:
            QQ = Q
        else:
            QQ = Q[t]

        u = Finv[t].dot(v[t]) - K[t].T.dot(r[t])
        thisr = ZZ.T.dot(u) + TT.T.dot(r[t])
        if t > 0:
            r[t - 1] = thisr
        else:
            r_init = thisr
            
    # run model forward
    alpha[0] = a_init + P_init.dot(r_init)
    RQR = RR.dot(QQ).dot(RR.T)

    for t in range(0, Ny - 1):
        alpha[t + 1] = TT.dot(alpha[t]) + RQR.dot(r[t])
        
    return alpha

def simulate(y, a_init, P_init, Z, H, T, R, Q):
    """
    Draw a sample of the state trajectory from the smoothed posterior.
    """
    # get dimensions:
    Ny, Np = y.shape
    Nm = Z.shape[1]
    Nr = Q.shape[0]
    
    # preallocate
    alpha_plus = np.empty((Ny, Nm))
    y_plus = np.empty((Ny, Np))
    
    # simulate data
    alpha_plus[0] = np.random.multivariate_normal(a_init, P_init)

    for t in range(Ny):
        if Z.ndim == 2:
            ZZ = Z
        else:
            ZZ = Z[t]

        if T.ndim == 2:
            TT = T
        else:
            TT = T[t]

        # draw disturbances
        eps = np.random.multivariate_normal(np.zeros(Np), H)
        eta = np.random.multivariate_normal(np.zeros(Nr), Q)
    
        y_plus[t] = ZZ.dot(alpha_plus[t]) + eps
        if t + 1 < Ny:
            alpha_plus[t + 1] = TT.dot(alpha_plus[t]) + R.dot(eta)
            
    # calculate smoothed means:
    
    # actual data
    v, K, Finv = kalman_filter(y, a_init, P_init, Z, H, T, R, Q)
    alpha_hat = fast_state_smoother(v, K, Finv, a_init, P_init, Z, T, R, Q)
    
    # simulated data
    v, K, Finv = kalman_filter(y_plus, a_init, P_init, Z, H, T, R, Q)
    alpha_hat_plus = fast_state_smoother(v, K, Finv, a_init, P_init, Z, T, R, Q)
    
    # combine
    alpha_draw = alpha_plus - alpha_hat_plus + alpha_hat
    
    return alpha_draw