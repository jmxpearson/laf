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